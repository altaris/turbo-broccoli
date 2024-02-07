# pylint: disable=import-outside-toplevel
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-builtin
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
"""
bokeh (de)serialization test suite. This test suite serializes and deserializes
various figures, but does not check for similarity between the original and the
deserialized...

TODO:
    Make every example of the [bokeh
    gallery](https://docs.bokeh.org/en/latest/docs/gallery.html) into a unit
    test. This is probably widely unnecessary though :]
"""

import numpy as np
from bokeh.core.enums import *
from bokeh.layouts import *
from bokeh.models import *
from bokeh.palettes import *
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.util.hex import hexbin
from common import to_from_json
from numpy.random import random, standard_normal
from scipy.special import jv
from scipy.stats import gaussian_kde


def test_markers():
    """https://docs.bokeh.org/en/latest/docs/examples/basic/scatters/markers.html"""
    p = figure(title="Bokeh Markers", toolbar_location=None)
    p.grid.grid_line_color = None
    p.background_fill_color = "#eeeeee"
    p.axis.visible = False
    p.y_range.flipped = True
    N = 10
    for i, marker in enumerate(MarkerType):
        x = i % 4
        y = (i // 4) * 4 + 1
        p.scatter(
            random(N) + 2 * x,
            random(N) + y,
            marker=marker,
            size=14,
            line_color="navy",
            fill_color="orange",
            alpha=0.5,
        )
        p.text(
            2 * x + 0.5,
            y + 2.5,
            text=[marker],
            text_color="firebrick",
            text_align="center",
            text_font_size="13px",
        )
    to_from_json(p)


def test_color_scatter():
    """https://docs.bokeh.org/en/latest/docs/examples/basic/scatters/color_scatter.html"""
    N = 4000
    x = np.random.random(size=N) * 100
    y = np.random.random(size=N) * 100
    radii = np.random.random(size=N) * 1.5
    cols = np.array(
        [(r, g, 150) for r, g in zip(50 + 2 * x, 30 + 2 * y)], dtype="uint8"
    )
    TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,examine,help"
    p = figure(tools=TOOLS)
    p.scatter(
        x, y, radius=radii, fill_color=cols, fill_alpha=0.6, line_color=None
    )
    to_from_json(p)


# def test_elements():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/scatters/elements.html"""
#     to_from_json(p)


# def test_image_url():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/scatters/image_url.html"""
#     to_from_json(p)


# def test_lorenz():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/lines/lorenz.html"""
#     to_from_json(p)


def test_linear_cmap():
    """https://docs.bokeh.org/en/latest/docs/examples/basic/data/linear_cmap.html"""
    x = standard_normal(50000)
    y = standard_normal(50000)
    bins = hexbin(x, y, 0.1)
    p = figure(tools="", match_aspect=True, background_fill_color="#440154")
    p.grid.visible = False
    p.hex_tile(
        q="q",
        r="r",
        size=0.1,
        line_color=None,
        source=bins,
        fill_color=linear_cmap("counts", "Viridis256", 0, max(bins.counts)),
    )
    to_from_json(p)


# def test_linear_cmap_colorbar():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/data/linear_cmap_colorbar.html"""
#     to_from_json(p)


# def test_color_mappers():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/data/color_mappers.html"""
#     to_from_json(p)


# def test_transform_markers():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/data/transform_markers.html"""
#     to_from_json(p)


# def test_transform_jitter():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/data/transform_jitter.html"""
#     to_from_json(p)


def test_logplot():
    """https://docs.bokeh.org/en/latest/docs/examples/basic/axes/logplot.html"""
    x = np.linspace(0.1, 5, 80)
    p = figure(
        title="log axis example",
        y_axis_type="log",
        x_range=(0, 5),
        y_range=(0.001, 10.0**22),
        background_fill_color="#fafafa",
    )
    p.line(
        x,
        np.sqrt(x),
        legend_label="y=sqrt(x)",
        line_color="tomato",
        line_dash="dashed",
    )
    p.line(x, x, legend_label="y=x")
    p.scatter(x, x, legend_label="y=x")
    p.line(x, x**2, legend_label="y=x**2")
    p.scatter(
        x,
        x**2,
        legend_label="y=x**2",
        fill_color=None,
        line_color="olivedrab",
    )
    p.line(x, 10**x, legend_label="y=10^x", line_color="gold", line_width=2)
    p.line(
        x,
        x**x,
        legend_label="y=x^x",
        line_dash="dotted",
        line_color="indigo",
        line_width=2,
    )
    p.line(
        x,
        10 ** (x**2),
        legend_label="y=10^(x^2)",
        line_color="coral",
        line_dash="dotdash",
        line_width=2,
    )
    p.legend.location = "top_left"
    to_from_json(p)


# def test_twin_axes():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/axes/twin_axes.html"""
#     to_from_json(p)


def test_fixed_axis():
    """https://docs.bokeh.org/en/latest/docs/examples/basic/axes/fixed_axis.html"""
    x = np.linspace(-6, 6, 500)
    y = 8 * np.sin(x) * np.sinc(x)
    p = figure(
        width=800,
        height=300,
        title="",
        tools="",
        toolbar_location=None,
        match_aspect=True,
    )
    p.line(x, y, color="navy", alpha=0.4, line_width=4)
    p.background_fill_color = "#efefef"
    p.xaxis.fixed_location = 0
    to_from_json(p)


# def test_basic():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/basic.html"""
#     to_from_json(p)


# def test_colormapped():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/colormapped.html"""
#     to_from_json(p)


# def test_intervals():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/intervals.html"""
#     to_from_json(p)


# def test_mixed():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/mixed.html"""
#     to_from_json(p)


# def test_nested_colormapped():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/nested_colormapped.html"""
#     to_from_json(p)


# def test_pandas_groupby_colormapped():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/pandas_groupby_colormapped.html"""
#     to_from_json(p)


# def test_pandas_groupby_nested():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/pandas_groupby_nested.html"""
#     to_from_json(p)


# def test_stacked():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/stacked.html"""
#     to_from_json(p)


# def test_stacked_split():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/stacked_split.html"""
#     to_from_json(p)


# def test_nested():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/nested.html"""
#     to_from_json(p)


# def test_colors():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/colors.html"""
#     to_from_json(p)


# def test_dodged():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/bars/dodged.html"""
#     to_from_json(p)


# def test_stacked_area():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/areas/stacked_area.html"""
#     to_from_json(p)


# def test_anscombe():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/layouts/anscombe.html"""
#     to_from_json(p)


# def test_legend():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/annotations/legend.html"""
#     to_from_json(p)


# def test_arrow():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/annotations/arrow.html"""
#     to_from_json(p)


# def test_band():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/annotations/band.html"""
#     to_from_json(p)


# def test_slope():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/annotations/slope.html"""
#     to_from_json(p)


# def test_span():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/annotations/span.html"""
#     to_from_json(p)


# def test_whisker():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/annotations/whisker.html"""
#     to_from_json(p)


# def test_colorbar_log():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/annotations/colorbar_log.html"""
#     to_from_json(p)


# def test_box_annotation():
#     """https://docs.bokeh.org/en/latest/docs/examples/basic/annotations/box_annotation.html"""
#     to_from_json(p)


# def test_grid_bounds():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/plots/grid_bounds.html"""
#     to_from_json(p)


# def test_minor_grid_lines():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/plots/minor_grid_lines.html"""
#     to_from_json(p)


# def test_grid_band_fill():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/plots/grid_band_fill.html"""
#     to_from_json(p)


# def test_hatch_grid_band():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/plots/hatch_grid_band.html"""
#     to_from_json(p)


# def test_glyph_selection():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/plots/glyph_selection.html"""
#     to_from_json(p)


# def test_glyph_hover():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/plots/glyph_hover.html"""
#     to_from_json(p)


# def test_legend_location_outside():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/plots/legend_location_outside.html"""
#     to_from_json(p)


# def test_legend_title():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/plots/legend_title.html"""
#     to_from_json(p)


# def test_latex_blackbody_radiation():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/mathtext/latex_blackbody_radiation.html"""
#     to_from_json(p)


# def test_latex_normal_distribution():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/mathtext/latex_normal_distribution.html"""
#     to_from_json(p)


# def test_latex_schrodinger():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/mathtext/latex_schrodinger.html"""
#     to_from_json(p)


def test_latex_bessel():
    """https://docs.bokeh.org/en/latest/docs/examples/styling/mathtext/latex_bessel.html"""
    p = figure(
        width=700,
        height=500,
        title=r"$$\color{white} \text{Bessel functions of the first kind: } J_\alpha(x) = \sum_{m=0}^{\infty}"
        r"\frac{(-1)^m}{m!\:\Gamma(m+\alpha+1)} \left(\frac{x}{2}\right)^{2m+\alpha}$$",
    )
    p.x_range.range_padding = 0
    p.xaxis.axis_label = r"$$\color{white} x$$"
    p.yaxis.axis_label = r"$$\color{white} J_\alpha(x)$$"
    p.title.text_font_size = "14px"
    x = np.linspace(0.0, 14.0, 100)
    for i, (xlabel, ylabel) in enumerate(
        zip([0.5, 1.6, 2.8, 4.2], [0.95, 0.6, 0.5, 0.45])
    ):
        p.line(x, jv(i, x), line_width=3, color=YlOrRd4[i])
        p.add_layout(
            Label(
                text=r"$$\color{white} J_" + str(i) + "(x)$$",
                x=xlabel,
                y=ylabel,
            )
        )
    to_from_json(p)


# def test_caliber():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/themes/caliber.html"""
#     to_from_json(p)


# def test_dark_minimal():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/themes/dark_minimal.html"""
#     to_from_json(p)


# def test_light_minimal():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/themes/light_minimal.html"""
#     to_from_json(p)


# def test_night_sky():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/themes/night_sky.html"""
#     to_from_json(p)


# def test_contrast():
#     """https://docs.bokeh.org/en/latest/docs/examples/styling/themes/contrast.html"""
#     to_from_json(p)


# def test_image_rgba():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/images/image_rgba.html"""
#     to_from_json(p)


# def test_image():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/images/image.html"""
#     to_from_json(p)


# def test_image_origin_anchor():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/images/image_origin_anchor.html"""
#     to_from_json(p)


def test_contour_simple():
    """https://docs.bokeh.org/en/latest/docs/examples/topics/contour/contour_simple.html"""
    x, y = np.meshgrid(np.linspace(0, 3, 40), np.linspace(0, 2, 30))
    z = 1.3 * np.exp(-2.5 * ((x - 1.3) ** 2 + (y - 0.8) ** 2)) - 1.2 * np.exp(
        -2 * ((x - 1.8) ** 2 + (y - 1.3) ** 2)
    )
    p = figure(width=550, height=300, x_range=(0, 3), y_range=(0, 2))
    levels = np.linspace(-1, 1, 9)
    cr = p.contour(x, y, z, levels, fill_color=Sunset8, line_color="black")
    colorbar = cr.construct_color_bar()
    p.add_layout(colorbar, "right")
    to_from_json(p)


# def test_contour():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/contour/contour.html"""
#     to_from_json(p)


# def test_contour_polar():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/contour/contour_polar.html"""
#     to_from_json(p)


# def test_hex_tile():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/hex/hex_tile.html"""
#     to_from_json(p)


# def test_hexbin():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/hex/hexbin.html"""
#     to_from_json(p)


# def test_ridgeplot():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/categorical/ridgeplot.html"""
#     to_from_json(p)


# def test_scatter_jitter():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/categorical/scatter_jitter.html"""
#     to_from_json(p)


# def test_les_mis():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/categorical/les_mis.html"""
#     to_from_json(p)


# def test_heatmap_unemployment():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/categorical/heatmap_unemployment.html"""
#     to_from_json(p)


# def test_periodic():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/categorical/periodic.html"""
#     to_from_json(p)


# def test_treemap():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/hierarchical/treemap.html"""
#     to_from_json(p)


# def test_crosstab():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/hierarchical/crosstab.html"""
#     to_from_json(p)


def test_texas_hover_map():
    """https://docs.bokeh.org/en/latest/docs/examples/topics/geo/texas_hover_map.html"""
    from bokeh.sampledata.unemployment import data as unemployment
    from bokeh.sampledata.us_counties import data as counties

    palette = tuple(reversed(Viridis6))
    counties = {
        code: county
        for code, county in counties.items()
        if county["state"] == "tx"
    }
    county_xs = [county["lons"] for county in counties.values()]
    county_ys = [county["lats"] for county in counties.values()]
    county_names = [county["name"] for county in counties.values()]
    county_rates = [unemployment[county_id] for county_id in counties]
    color_mapper = LogColorMapper(palette=palette)
    data = {
        "x": county_xs,
        "y": county_ys,
        "name": county_names,
        "rate": county_rates,
    }
    TOOLS = "pan,wheel_zoom,reset,hover,save"
    p = figure(
        title="Texas Unemployment, 2009",
        tools=TOOLS,
        x_axis_location=None,
        y_axis_location=None,
        tooltips=[
            ("Name", "@name"),
            ("Unemployment rate", "@rate%"),
            ("(Long, Lat)", "($x, $y)"),
        ],
    )
    p.grid.grid_line_color = None
    p.hover.point_policy = "follow_mouse"
    p.patches(
        "x",
        "y",
        source=data,
        fill_color={"field": "rate", "transform": color_mapper},
        fill_alpha=0.7,
        line_color="white",
        line_width=0.5,
    )
    to_from_json(p)


# def test_eclipse():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/geo/eclipse.html"""
#     to_from_json(p)


# def test_tile_source():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/geo/tile_source.html"""
#     to_from_json(p)


# def test_tile_xyzservices():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/geo/tile_xyzservices.html"""
#     to_from_json(p)


# def test_tile_demo():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/geo/tile_demo.html"""
#     to_from_json(p)


# def test_gmap():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/geo/gmap.html"""
#     to_from_json(p)


# def test_from_networkx():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/graph/from_networkx.html"""
#     to_from_json(p)


# def test_node_and_edge_attributes():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/graph/node_and_edge_attributes.html"""
#     to_from_json(p)


# def test_candlestick():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/timeseries/candlestick.html"""
#     to_from_json(p)


# def test_missing_dates():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/timeseries/missing_dates.html"""
#     to_from_json(p)


# def test_pie():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/pie/pie.html"""
#     to_from_json(p)


# def test_donut():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/pie/donut.html"""
#     to_from_json(p)


# def test_burtin():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/pie/burtin.html"""
#     to_from_json(p)


# def test_histogram():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/stats/histogram.html"""
#     to_from_json(p)


def test_kde2d():
    """https://docs.bokeh.org/en/latest/docs/examples/topics/stats/kde2d.html"""
    from bokeh.sampledata.autompg import autompg as df

    def kde(x, y, N):
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        X, Y = np.mgrid[xmin : xmax : N * 1j, ymin : ymax : N * 1j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        return X, Y, Z

    x, y, z = kde(df.hp, df.mpg, 300)
    p = figure(
        height=400,
        x_axis_label="hp",
        y_axis_label="mpg",
        background_fill_color="#fafafa",
        tools="",
        toolbar_location=None,
        title="Kernel density estimation plot of HP vs MPG",
    )
    p.grid.level = "overlay"
    p.grid.grid_line_color = "black"
    p.grid.grid_line_alpha = 0.05
    palette = Blues9[::-1]
    levels = np.linspace(np.min(z), np.max(z), 10)
    p.contour(x, y, z, levels[1:], fill_color=palette, line_color=palette)
    to_from_json(p)


# def test_splom():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/stats/splom.html"""
#     to_from_json(p)


# def test_boxplot():
#     """https://docs.bokeh.org/en/latest/docs/examples/topics/stats/boxplot.html"""
#     to_from_json(p)


def test_range_tool():
    """https://docs.bokeh.org/en/latest/docs/examples/interaction/tools/range_tool.html"""
    from bokeh.sampledata.stocks import AAPL

    dates = np.array(AAPL["date"], dtype=np.datetime64)
    source = ColumnDataSource(data={"date": dates, "close": AAPL["adj_close"]})
    p = figure(
        height=300,
        width=800,
        tools="xpan",
        toolbar_location=None,
        x_axis_type="datetime",
        x_axis_location="above",
        background_fill_color="#efefef",
        x_range=(dates[1500], dates[2500]),
    )
    p.line("date", "close", source=source)
    p.yaxis.axis_label = "Price"
    select = figure(
        title="Drag the middle and edges of the selection box to change the range above",
        height=130,
        width=800,
        y_range=p.y_range,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
    )
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line("date", "close", source=source)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    p = column(p, select)
    to_from_json(p)


# def test_linked_brushing():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/linking/linked_brushing.html"""
#     to_from_json(p)


# def test_linked_crosshair():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/linking/linked_crosshair.html"""
#     to_from_json(p)


# def test_data_table_plot():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/linking/data_table_plot.html"""
#     to_from_json(p)


# def test_legend_hide():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/legends/legend_hide.html"""
#     to_from_json(p)


# def test_legend_mute():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/legends/legend_mute.html"""
#     to_from_json(p)


def test_slider():
    """https://docs.bokeh.org/en/latest/docs/examples/interaction/js_callbacks/slider.html"""
    x = np.linspace(0, 10, 500)
    y = np.sin(x)
    source = ColumnDataSource(data={"x": x, "y": y})
    plot = figure(y_range=(-10, 10), width=400, height=400)
    plot.line("x", "y", source=source, line_width=3, line_alpha=0.6)
    amp = Slider(start=0.1, end=10, value=1, step=0.1, title="Amplitude")
    freq = Slider(start=0.1, end=10, value=1, step=0.1, title="Frequency")
    phase = Slider(start=-6.4, end=6.4, value=0, step=0.1, title="Phase")
    offset = Slider(start=-9, end=9, value=0, step=0.1, title="Offset")
    callback = CustomJS(
        args={
            "source": source,
            "amp": amp,
            "freq": freq,
            "phase": phase,
            "offset": offset,
        },
        code="""
        const A = amp.value
        const k = freq.value
        const phi = phase.value
        const B = offset.value

        const x = source.data.x
        const y = Array.from(x, (x) => B + A*Math.sin(k*x+phi))
        source.data = { x, y }
    """,
    )
    amp.js_on_change("value", callback)
    freq.js_on_change("value", callback)
    phase.js_on_change("value", callback)
    offset.js_on_change("value", callback)
    p = row(plot, column(amp, freq, phase, offset))
    to_from_json(p)


# def test_color_sliders():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/js_callbacks/color_sliders.html"""
#     to_from_json(p)


def test_customjs_lasso_mean():
    """https://docs.bokeh.org/en/latest/docs/examples/interaction/js_callbacks/customjs_lasso_mean.html"""
    x = [random() for x in range(500)]
    y = [random() for y in range(500)]
    color = ["navy"] * len(x)
    s = ColumnDataSource(data={"x": x, "y": y, "color": color})
    p = figure(
        width=400, height=400, tools="lasso_select", title="Select Here"
    )
    p.circle(
        "x",
        "y",
        color="color",
        size=8,
        source=s,
        alpha=0.4,
        selection_color="firebrick",
    )
    s2 = ColumnDataSource(data={"x": [0, 1], "ym": [0.5, 0.5]})
    p.line(x="x", y="ym", color="orange", line_width=5, alpha=0.6, source=s2)
    s.selected.js_on_change(
        "indices",
        CustomJS(
            args={"s": s, "s2": s2},
            code="""
        const inds = s.selected.indices
        if (inds.length > 0) {
            const ym = inds.reduce((a, b) => a + s.data.y[b], 0) / inds.length
            s2.data = { x: s2.data.x, ym: [ym, ym] }
        }
    """,
        ),
    )
    to_from_json(p)


# def test_js_on_event():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/js_callbacks/js_on_event.html"""
#     to_from_json(p)


# def test_multiselect():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/widgets/multiselect.html"""
#     to_from_json(p)


# def test_multichoice():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/widgets/multichoice.html"""
#     to_from_json(p)


# def test_date_picker():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/widgets/date_picker.html"""
#     to_from_json(p)


# def test_dropdown():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/widgets/dropdown.html"""
#     to_from_json(p)


# def test_data_table():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/widgets/data_table.html"""
#     to_from_json(p)


# def test_data_cube():
#     """https://docs.bokeh.org/en/latest/docs/examples/interaction/widgets/data_cube.html"""
#     to_from_json(p)
