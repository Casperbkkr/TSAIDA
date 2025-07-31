from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import HoverTool, CrosshairTool, ColumnDataSource, LinearColorMapper, Scatter
from bokeh.palettes import tol as sp
import numpy as np

def plot(df_values, df_results, pred=0):
    date_dict = {"date": df_values['timestamp']}
    res_dict = {col: df_results[col] for col in df_results.columns}
    val_dict = {col: df_values[col] for col in df_values.columns}
    pred_dict = {"pred": pred}
    data = res_dict | val_dict
    data = data | date_dict
    data = data | pred_dict
    data = ColumnDataSource(data)

    tool_tips_top = [
        ("Time", "@date"),
        ("Value", "$y")]


    top = figure(height=300, width=1000,
                 #x_axis_type="datetime",
                 x_axis_location="above",
                 background_fill_color="beige", tooltips=tool_tips_top,
                 title="Data")

    tool_tips_top = [
        ("Time", "@date_print"),
        ("Value", "$y")]
    """
    top.line(x="date", y="heart_rate", source=data, line_color="blue", line_width=2)
    top.line(x="date", y="enhanced_speed", source=data, line_color="green", line_width=2)
    top.line(x="date", y="distance", source=data, line_color="blue", line_width=2)
    #top.line(x="date", y="position_long", source=data, line_color="red", line_width=2)
    """

    top.line(x="date", y="channel_12", source=data, line_color="blue", line_width=2)
    top.line(x="date", y="channel_13", source=data, line_color="green", line_width=2)
    top.line(x="date", y="channel_14", source=data, line_color="blue", line_width=2)
    top.line(x="date", y="channel_15", source=data, line_color="red", line_width=2)
    top.line(x="date", y="channel_16", source=data, line_color="red", line_width=2)
    top.line(x="date", y="channel_17", source=data, line_color="red", line_width=2)
    top.line(x="date", y="channel_18", source=data, line_color="red", line_width=2)
    top.line(x="date", y="channel_19", source=data, line_color="red", line_width=2)
    top.scatter(x="date", y="pred", source=data, size=2)
    top.scatter(x="date", y="anom", source=data, color="red", size=2)
    #top.grid.grid_line_width = 2
    top.add_tools(HoverTool(mode='vline', tooltips=tool_tips_top))
    top.add_tools(CrosshairTool(dimensions="height"))

    #Make bottom figure
    bottom = figure(height=300, width=1000,
                    x_axis_type="datetime",
                    x_axis_location="below",
                    background_fill_color="beige",
                    title="Outlier Scores")

    tool_tips_bottom = [
        ("Time", "@date_print"),
        ("Score Fourier", "@out0"),
        ("Score Signature", "@out1"),
        ("Score no transform", "@out2")
    ]


    bottom.add_tools(HoverTool(mode='vline', tooltips=tool_tips_bottom))
    bottom.add_tools(CrosshairTool(dimensions="height"))

    bottom.line(x="date", y="out1",legend_label="Distance", source=data, line_color="blue", line_width=1)
    #bottom.line(x="date", y="out2", source=data, line_color="red", line_width=1)
    bottom.line(x="date", y="out3",legend_label="Signature", source=data, line_color="green", line_width=1)


    show(column(top, bottom))


"""
def plot(df_plot):

    x_0 = df_plot['timestamp'].to_numpy()[0]
    y_0 = df_plot['value'].to_numpy().min(axis=0)-0.5
    x_range = (df_plot['timestamp'].to_numpy()[0], df_plot['timestamp'].to_numpy()[-1])
    y_range_top = (y_0, 1.1*df_plot['value'].to_numpy().max(axis=0))

    data = {
        'date': df_plot['timestamp'].to_numpy(),
        'date_print': (df_plot['timestamp'].astype('str')).to_numpy(),
        'value': df_plot['value'],
        'out0': df_plot["out0"],
        'out1': df_plot["out1"],
        'out2': df_plot["out2"]
    }

    data = ColumnDataSource(data)

    tool_tips_top = [
        ("Time", "@date"),
        ("Value", "$y")]


    background_value = df_plot['out1'].to_numpy()
    plotArray = np.repeat(background_value[:, np.newaxis], axis=1, repeats=256).transpose()



    top = figure(height=300, width=1000,
                 x_range=x_range,
                 y_range=y_range_top,
                 x_axis_type="datetime",
                 x_axis_location="above",
                 background_fill_color="beige", tooltips=tool_tips_top,
                 title="Data")



    tool_tips_top = [
        ("Time", "@date_print"),
        ("Value", "$y")]

    # Make gradient background
    pal = sp["Iridescent"][23]
    pal = "Magma256"
    color_mapper = LinearColorMapper(palette=pal,
                                     low=df_plot['out1'].to_numpy().min(axis=0) - 0.5,
                                     high=df_plot['out1'].to_numpy().max(axis=0))
    top.line(x="date", y="value", source=data, line_color="blue", line_width=2)

    top.image(image=[plotArray],
              color_mapper=color_mapper,
              dw=df_plot['timestamp'].to_numpy()[-1] - df_plot['timestamp'].to_numpy()[0],
              dh=df_plot['value'].to_numpy().max(axis=0) + 1,
              x=x_0,
              y=y_0,
              level="image")


    #top.grid.grid_line_width = 2
    top.add_tools(HoverTool(mode='vline', tooltips=tool_tips_top))
    top.add_tools(CrosshairTool(dimensions="height"))

    #Make bottom figure
    bottom = figure(height=300, width=1000,
                    x_axis_type="datetime",
                    x_axis_location="below",
                    x_range=x_range,
                    background_fill_color="beige",
                    title="Outlier Scores")

    tool_tips_bottom = [
        ("Time", "@date_print"),
        ("Score Fourier", "@out0"),
        ("Score Signature", "@out1"),
        ("Score no transform", "@out2")
    ]


    bottom.add_tools(HoverTool(mode='vline', tooltips=tool_tips_bottom))
    bottom.add_tools(CrosshairTool(dimensions="height"))

    bottom.line(x="date", y="out2", source=data, legend_label="No transform", line_color="blue", line_width=1)
    bottom.line(x="date", y="out0", source=data, legend_label="Fourier", line_color="red", line_width=1)
    bottom.line(x="date", y="out1", source=data, legend_label="Signature", line_color="green", line_width=1)

    # Add a segment to the plot

    show(column(top, bottom))
    """
