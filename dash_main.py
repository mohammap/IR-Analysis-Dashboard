import dash
from dash import html
import pandas as pd
from dash import dcc
from dash.dependencies import Input, Output
from dash import dash_table
import plotly
import base64
from flask import Flask, request
from dash import Dash
from dash import dcc
import werkzeug


df = pd.read_csv('masterfile_demo.csv', encoding= 'unicode_escape')
df_1 = pd.read_csv('masterfile_demo_part.csv', encoding= 'unicode_escape')

image_filename ='plots/part_dynamic/demo_part_dynamic.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

image_filename2 = 'plots/homline_part/demo_hom_line_part.png' # replace with your own image
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())

image_filename3 = 'plots/cropped_tool_with_temp_mm/cropped_tool_with_temp_demo.png' # replace with your own image
encoded_image3 = base64.b64encode(open(image_filename3, 'rb').read())

image_filename4 = 'plots/histograms_with_distributions/histo_dist_demo.png' # replace with your own image
encoded_image4 = base64.b64encode(open(image_filename4, 'rb').read())

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H4(children='Thermographic analysis of the consolidation press tool', style={'textAlign': 'center', 'fontSize' : 20}),

    html.Img(
        id='image3',
        src='data:image/png;base64,{}'.format(encoded_image.decode()),
        height=280),

    html.Img(
        id='image4',
        src='data:image/png;base64,{}'.format(encoded_image2.decode()),
        height=280),

    html.Div(html.P([ html.Br()])),

    html.Img(
        id='image1',
        src='data:image/png;base64,{}'.format(encoded_image2.decode()),
        height=280),

    html.Img(
        id='image2',
        src='data:image/png;base64,{}'.format(encoded_image2.decode()),
        height=280),
    html.H4(children='Statistical summary of the tool temperature',style={'fontSize' : 16}),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
    ),
    html.H4(children='Statistical summary of the part area temperature',style={'fontSize' : 16}),
    dash_table.DataTable(
        id='table1',
        columns=[{"name": i, "id": i} for i in df_1.columns],
        data=df_1.to_dict('records'),
    ),
    dcc.Interval(
            id='interval-component',
            interval=1000, # in milliseconds
            n_intervals=0
        )

])

#  html.Br(),
@app.callback(
        dash.dependencies.Output('table','data'),
        [dash.dependencies.Input('interval-component', 'n_intervals')])
def updateTable(n):
    df = pd.read_csv('masterfile_demo.csv')
    return df.to_dict('records')

@app.callback(
        dash.dependencies.Output('table1','data'),
        [dash.dependencies.Input('interval-component', 'n_intervals')])
def updateTable(n):
    df_1 = pd.read_csv('masterfile_demo_part.csv')
    return df_1.to_dict('records')

@app.callback(
        dash.dependencies.Output('image1','src'),
        [dash.dependencies.Input('interval-component', 'n_intervals')])
def updateImage1(n):
    image_filename = 'plots/part_dynamic/demo_part_dynamic.png'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())

@app.callback(
        dash.dependencies.Output('image2','src'),
        [dash.dependencies.Input('interval-component', 'n_intervals')])
def updateImage1(n):
    image_filename = 'plots/homline_part/demo_hom_line_part.png'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())

@app.callback(
        dash.dependencies.Output('image3','src'),
        [dash.dependencies.Input('interval-component', 'n_intervals')])
def updateImage1(n):
    image_filename = 'plots/cropped_tool_with_temp_mm/cropped_tool_with_temp_demo.png'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())

@app.callback(
        dash.dependencies.Output('image4','src'),
        [dash.dependencies.Input('interval-component', 'n_intervals')])
def updateImage1(n):
    image_filename = 'plots/histograms_with_distributions/histo_dist_demo.png'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


# if __name__ == '__main__':
#     app.run_server(debug=True)


#integrating flask application with dash:

    server = Flask(__name__)
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname='/dash'
    )

    app.layout = html.Div(id='pariush') \
 \
                 @ server.route("/dash")

    def my_dash_app():
        return app.index()



    #path = r'C:\Users\AK125163\Desktop\zanosh1'  # use your path
    #all_files = glob.glob(path + "/*.csv")

    #li = []

    # for filename in all_files:
    #     df = pd.read_csv(filename, index_col=None, header=0)
    #     li.append(df)
    #
    # frame = pd.concat(li, axis=0, ignore_index=True)



    @app.route('/upload', methods=['POST'])
    def upload():
          df = pd.read_csv(request.files.get('uploaded_file'))
if __name__ == "__main__":

          app.run(host="0.0.0.0", port=8050, debug=False)

# # Py file name: dash_main.py
# app = Flask(__name__)
# api = Api(app)
#
# class kzwebfile(Resource):
#
#     def post(self, dash_main):
#         pass
#
# api.add_resource(kzwebfile, '/<string:dash_main>')
#
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=8050, debug=False)