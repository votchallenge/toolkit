from typing import Tuple

import importlib
import multiprocessing
import queue
import traceback
import inspect
import time

from vot.dataset import Frame
from vot.tracker import Tracker, OnlineTrackerRuntime, FrameObjects, TrackerException


def _resolve_factory(command: str):
    if ":" in command:
        module_name, object_name = command.split(":", 1)
    elif "." in command:
        module_name, object_name = command.rsplit(".", 1)
    else:
        module_name, object_name = command, None

    module = importlib.import_module(module_name)

    if object_name is None:
        if hasattr(module, "create_tracker"):
            return getattr(module, "create_tracker")
        if hasattr(module, "Tracker"):
            return getattr(module, "Tracker")
        raise RuntimeError("Unable to resolve tracker factory from command '{}'".format(command))

    if not hasattr(module, object_name):
        raise RuntimeError("Object '{}' not found in module '{}'".format(object_name, module_name))

    return getattr(module, object_name)


def _call_tracker_method(method, frame, new, properties):
    signature = inspect.signature(method)
    arity = len(signature.parameters)

    if arity >= 3:
        return method(frame, new, properties)
    if arity == 2:
        return method(frame, new)
    if arity == 1:
        return method(frame)
    return method()


def _worker_main(command, task_queue, result_queue, arguments, initialize_method, update_method):
    try:
        factory = _resolve_factory(command)
        tracker = factory(**(arguments or {})) if callable(factory) else factory

        if hasattr(tracker, initialize_method):
            init_name = initialize_method
        elif hasattr(tracker, "init"):
            init_name = "init"
        elif hasattr(tracker, "initialize"):
            init_name = "initialize"
        else:
            raise RuntimeError("Tracker object does not expose an initialization method")

        if not hasattr(tracker, update_method):
            raise RuntimeError("Tracker object does not expose '{}' method".format(update_method))

        init_fn = getattr(tracker, init_name)
        update_fn = getattr(tracker, update_method)

        result_queue.put({
            "ok": True,
            "event": "ready",
            "multiobject": bool(getattr(tracker, "multiobject", True)),
            "initialize_method": init_name,
            "update_method": update_method
        })

        while True:
            task = task_queue.get()
            task_type = task.get("type")

            if task_type == "stop":
                result_queue.put({"ok": True, "event": "stopped"})
                break

            frame = task.get("frame")
            new = task.get("new")
            properties = task.get("properties")

            start = time.time()

            if task_type == "initialize":
                output = _call_tracker_method(init_fn, frame, new, properties)
            elif task_type == "update":
                output = _call_tracker_method(update_fn, frame, new, properties)
            else:
                raise RuntimeError("Unknown task type '{}'".format(task_type))

            elapsed = time.time() - start

            if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], (int, float)):
                status = output[0]
                elapsed = float(output[1])
            else:
                status = output

            result_queue.put({"ok": True, "event": task_type, "status": status, "time": elapsed})

    except (RuntimeError, TypeError, ValueError, ImportError, AttributeError) as e:
        result_queue.put({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


class PythonRuntime(OnlineTrackerRuntime):
    """Multiprocessing runtime for Python-native trackers.

    The command is interpreted as an import path to a tracker factory/class,
    for example: ``mypackage.mytracker:Tracker``.
    """

    def __init__(self, tracker: Tracker, command: str, log: bool = False, timeout: int = 30, linkpaths=None, envvars=None, arguments=None, **kwargs):
        super().__init__(tracker)

        self._command = command
        self._timeout = timeout
        self._log = log
        self._linkpaths = linkpaths
        self._envvars = envvars
        self._arguments = arguments if arguments is not None else {}
        self._initialize_method = kwargs.get("initialize_method", "init")
        self._update_method = kwargs.get("update_method", "update")

        self._task_queue = None
        self._result_queue = None
        self._process = None
        self._multiobject = kwargs.get("multiobject", True)

    def _timeout_value(self):
        return None if self._timeout is None or self._timeout <= 0 else self._timeout

    def _wait_message(self):
        try:
            return self._result_queue.get(timeout=self._timeout_value())
        except queue.Empty as e:
            raise TrackerException(
                "Python tracker runtime timed out after {} seconds".format(self._timeout),
                tracker=self.tracker
            ) from e

    def _raise_worker_error(self, message):
        details = message.get("traceback")
        error = message.get("error", "Unknown worker error")
        raise TrackerException(
            "Python tracker worker failed: {}".format(error),
            tracker=self.tracker,
            tracker_log=details
        )

    def _ensure_started(self):
        if self._process is not None and self._process.is_alive():
            return

        context = multiprocessing.get_context("spawn")
        self._task_queue = context.Queue()
        self._result_queue = context.Queue()
        self._process = context.Process(
            target=_worker_main,
            args=(
                self._command,
                self._task_queue,
                self._result_queue,
                self._arguments,
                self._initialize_method,
                self._update_method
            )
        )
        self._process.start()

        message = self._wait_message()
        if not message.get("ok", False):
            self.stop()
            self._raise_worker_error(message)

        if message.get("event") != "ready":
            self.stop()
            raise TrackerException("Python tracker worker did not enter ready state", tracker=self.tracker)

        self._multiobject = bool(message.get("multiobject", True))

    def _send_task(self, task_type, frame, new=None, properties=None) -> Tuple[FrameObjects, float]:
        self._ensure_started()

        payload = {
            "type": task_type,
            "frame": frame,
            "new": [] if new is None else new,
            "properties": {} if properties is None else properties
        }

        self._task_queue.put(payload)
        message = self._wait_message()

        if not message.get("ok", False):
            self.stop()
            self._raise_worker_error(message)

        if message.get("event") != task_type:
            self.stop()
            raise TrackerException(
                "Unexpected worker event '{}' while waiting for '{}'".format(message.get("event"), task_type),
                tracker=self.tracker
            )

        return message.get("status", []), float(message.get("time", 0.0))

    def initialize(self, frame: Frame, new: FrameObjects = None, properties: dict = None) -> Tuple[FrameObjects, float]:
        if not self.multiobject and new is not None and len(new) > 1:
            raise TrackerException(
                "Tracker does not support multiple objects, but multiple objects were provided for initialization",
                tracker=self.tracker
            )

        return self._send_task("initialize", frame, new, properties)

    def update(self, frame: Frame, new: FrameObjects = None, properties: dict = None) -> Tuple[FrameObjects, float]:
        if not self.multiobject and new is not None and len(new) > 0:
            raise TrackerException(
                "Tracker does not support multiple objects, but multiple objects were provided for update",
                tracker=self.tracker
            )

        return self._send_task("update", frame, new, properties)

    def restart(self):
        self.stop()
        self._ensure_started()

    def stop(self):
        if self._process is None:
            return

        try:
            if self._process.is_alive() and self._task_queue is not None:
                self._task_queue.put({"type": "stop"})
                self._process.join(timeout=2)
        except (OSError, ValueError, EOFError, BrokenPipeError):
            pass

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)

        try:
            if self._task_queue is not None:
                self._task_queue.close()
            if self._result_queue is not None:
                self._result_queue.close()
        except (OSError, ValueError):
            pass

        self._task_queue = None
        self._result_queue = None
        self._process = None

    @property
    def multiobject(self):
        return self._multiobject
    
from vot.tracker import register_runtime_protocol

register_runtime_protocol("python", PythonRuntime)
