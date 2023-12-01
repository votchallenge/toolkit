
function get_tracker_id(el) {
    var id = $(el).attr("id");
    var index = id.lastIndexOf("_");
    return parseInt(id.substring(index+1));
}

function update_selected(el) {
    
    var selected = $("tr.selected");
    if (selected.length == 0) {
        $("g.tracker").removeClass("blurred");
    } else {
        $("g.tracker").addClass("blurred");

        selected.each(function (i, el) {
            var tracker = $(el).data("tracker");
            $("g.tracker_" + tracker).removeClass("blurred");
        });
    }

}

$(function() {

    var trackerNames={};

    $("td[id^='legend_']").each(function (i, el) {
        var tracker = get_tracker_id(el);
        $(el).addClass("tracker");
        trackerNames[tracker] = $(el).children("span").text();
        $(el).parent().click(function () {
            $(this).toggleClass("selected");
            update_selected();
        }).data("tracker", tracker);
    });

    $("g[id^='report_']").each(function (i, el) {
        var tracker = get_tracker_id(el);
        $(el).addClass("tracker tracker_" + tracker);
        $(el).find("path").after($("<title>").text(trackerNames[tracker]));
    });

    $("table.overview-table").stupidtable();





});