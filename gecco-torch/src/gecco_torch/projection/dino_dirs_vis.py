import plotly.graph_objects as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

class Visualize_cameras():

    def __init__(self,port,origins,dirs):
        self.port = port
        self.set_layout()
        app = dash.Dash(__name__)
        self.bool_positive_z_axis_viewing_direction = False

        self.origins, self.dirs = origins, dirs
        app.layout = html.Div([
            html.Button('Swap X and Y axis',id="swapxy",n_clicks=0),
            html.Button('Swap X and Z axis',id="swapxz",n_clicks=0),
            html.Button('Swap Y and Z axis',id="swapyz",n_clicks=0),
            html.Button('Rotate around X',id="rotatex",n_clicks=0),
            html.Button('Rotate around Y',id="rotatey",n_clicks=0),
            html.Button('Rotate around Z',id="rotatez",n_clicks=0),
            html.Button('Change viewing direction along all axis',id="changeallaxis",n_clicks=0),
            html.Button('Click for next pose',id="next_pose",n_clicks=0),
            html.Button('Click for prev pose',id="prev_pose",n_clicks=0),
            html.P(id="output-text1", children="Loading..."),
            html.P(id="output-text2", children="Loading..."),
            html.P(id="output-text3", children="Loading..."),
            dcc.Graph(id='cams_view',clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip"),
        ])
        @app.callback(
            Output("output-text1", "children"),
            Output("output-text2", "children"),
            Output("output-text3", "children"),
            Output('cams_view', 'figure'),
            Input('swapxy', 'n_clicks'),
            Input('swapxz', 'n_clicks'),
            Input('swapyz', 'n_clicks'),
            Input('rotatex', 'n_clicks'),
            Input('rotatey', 'n_clicks'),
            Input('rotatez', 'n_clicks'),
            Input('changeallaxis', 'n_clicks'),
            Input('next_pose', 'n_clicks'),
            Input('prev_pose', 'n_clicks'),
        )
        def update_3d_plot(number_swapxy,number_swapxz,number_swapyz,number_rotatex,number_rotatey,number_rotatez,number_changeallaxis,number_pose_forw,number_pose_back):
            rotation_text = "Rotation around X: {}, Rotation around Y: {}, Rotation around Z: {}".format(90*(number_rotatex%4),90*(number_rotatey%4),90*(number_rotatez%4))
            info_text = ""
            if number_changeallaxis % 2 != 0:
                # world_matrices = change_all_axis_viewing_direction(world_matrices)
                self.bool_positive_z_axis_viewing_direction = True
            else:
                self.bool_positive_z_axis_viewing_direction = False
            info_text += "Changed viewingdirections: {}".format("Yes" if number_changeallaxis%2 != 0 else "No") + ". "

            self.pose_idx = (number_pose_forw-number_pose_back) % (len(self.origins)+1)
            # self.pose_idx = number_pose_forw % len(self.world_matrices)
            print(self.origins)
            print(self.dirs)
            if self.pose_idx == 0:
                # all
                pose_info_text = "Showing all poses"
                fig = self.get_fig_camera_arrows(self.origins,self.dirs)
            else:
                fig = self.get_fig_camera_arrows([self.origins[self.pose_idx-1]], [self.dirs[self.pose_idx-1]])
                pose_info_text = "Showing pose nr. {}".format(self.pose_idx-1)
            print("HERE")
            fig.update_traces(hoverinfo="none", hovertemplate=None) # hide the little window with the coordinates
            return pose_info_text,info_text,rotation_text,fig
        
        self.app = app

        
        
    def run(self):
        self.app.run_server(debug=True,port=self.port)

    # def set_layout(self):
    #     max_val = 3.4
    #     self.layout = go.Layout(
    #         margin=dict(l=0, r=0, b=0, t=0),
    #         scene=dict(
    #             xaxis=dict(
    #                 # autorange='reversed',  # This reverses the x-axis
    #                 range=[max_val,-max_val],
    #                 autorange=False
    #             ),
    #             yaxis=dict(
    #                 # autorange='reversed',  # This reverses the y-axis
    #                 range=[max_val,-max_val],
    #                 autorange=False
    #             ),
    #             zaxis = dict(
    #                 range=[-max_val,max_val],
    #                 autorange=False
    #             ),
    #             aspectratio=dict(x=1, y=1, z=1),
    #         )
    #     )
    def set_layout(self):
        max_val = 3.4
        self.layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    range=[max_val,-max_val],
                    autorange=False
                ),
                yaxis=dict(
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    range=[max_val,-max_val],
                    autorange=False
                ),
                zaxis=dict(
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    range=[-max_val,max_val],
                    autorange=False
                ),
                aspectratio=dict(x=1, y=1, z=1),
            )
        )

    
    
    def get_fig_camera_arrows(self,origins,dirs):
        self.origins,self.dirs = origins,dirs
        # for i,o in enumerate(dirs):
        #     print(i)
        #     print(o)
        visual_elements = []
        for o,d in zip(origins,dirs):
            d = d/3
            # Define the base vectors
            # print("start: ",o)
            # print("end: ",o+d)
            base_vector = go.Scatter3d(
                x=[o[0], o[0] + d[0]],
                y=[o[1], o[1] + d[1]],
                z=[o[2], o[2] + d[2]],
                marker=dict(size=1, color="rgb(84,48,5)"),
                line=dict(color="rgb(84,48,5)", width=6)
            )

            visual_elements.append(base_vector)
            arrowhead = go.Cone(
                x=[o[0] + d[0]],
                y=[o[1] + d[1]],
                z=[o[2] + d[2]],
                u=[d[0]],  # Modify the arrowhead size and direction as needed
                v=[d[1]],
                w=[d[2]],
                showscale=False,  # Hide color scale for arrowhead
                sizemode="absolute",  # Use an absolute size for arrowhead
                sizeref=0.3,  # Adjust arrowhead size
            )
            visual_elements.append(arrowhead)
        fig = go.Figure(data=visual_elements, layout=self.layout)
        return fig
    
if __name__ == "__main__":
    np_dict = np.load("camdirs.npz")
    cam = np_dict["cam"]
    dirs = np_dict["dirs"]
    cam = cam[0].reshape(-1,3)
    dirs = dirs[0].reshape(-1,3)
    cam_indices = np.random.randint(0,len(cam),size=100)
    cam = cam[cam_indices]
    dirs = dirs[cam_indices]
    # dirs = 4 * dirs / np.linalg.norm(dirs,axis=1)[:,None]
    vc = Visualize_cameras(5050,cam, dirs)
    vc.run()