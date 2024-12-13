import collections

import numpy as np
import plotly.graph_objects as go


class EteePoseVisualisation:
    def __init__(self, body):
        """
        Initialize the body part indices and check for mat corner.
        """
        self.body = body
        self.axis = collections.defaultdict(int)
        self.mat_corner = len(body) > 3  # Check if there are mat corners (if the body has 4 parts)

    def __call__(self, keypoint=None, mat=None, gt=None):
        """
        The main entry point for visualizing the body pose.
        keypoint: Pose keypoints (coordinates for the body parts)
        mat: Pressure map data (optional)
        gt: Ground truth data (optional, not currently used)
        """
        self.idx = len(keypoint)  # Number of frames
        self.set_frames()

        body_data = self.set_body_data(keypoint)  # Set body part data (arm, leg, torso)
        self.update_max_axis(keypoint)
        self.add_body_frames(body_data)  # Create frames for animation


        if self.mat_corner:
            self.set_mat_corner_data(keypoint)
            self.add_mat_corner_frames()

        if mat is not None:
            self.set_pressure_map_data(mat)
            self.add_pressure_map_frames()


        if gt is not None:
            body_data_gt = self.set_body_data(gt)  # Set body part data (arm, leg, torso)
            self.update_max_axis(keypoint)
            self.add_body_frames(body_data_gt, color=["black"] * 3)  # Create frames for ground truth data

        # Add initial traces
        self._add_initial_traces(body_data, color=['black', 'red', 'blue'])
        if self.mat_corner:
            self.fig.add_trace(self._create_trace(self.mat_tracker_data, 0, color='green'))
        if mat is not None:
            self.fig.add_trace(
                go.Surface(x=self.mat_x, y=self.mat_y, z=self.pressure_map_data[0], cmin=0, cmax=100))
        if gt is not None:
            self._add_initial_traces(body_data_gt, color=['black', 'red', 'blue'])
        self.plot()

    def update_max_axis(self, data):
        """
        Find the max and min x, y, z axis in keypoint data
        """
        if data is None:
            None
        else:
            self.axis["x_min"] = min(self.axis["x_min"], np.min(data[:,:,0]))
            self.axis["y_min"] = min(self.axis["y_min"], np.min(data[:, :, 1]))
            self.axis["z_min"] = min(self.axis["z_min"], np.min(data[:, :, 2]))
            self.axis["x_max"] = max(self.axis["x_max"], np.max(data[:, :, 0]))
            self.axis["y_max"] = max(self.axis["y_max"], np.max(data[:, :, 1]))
            self.axis["z_max"] = max(self.axis["z_max"], np.max(data[:, :, 2]))

    def set_body_data(self, keypoint):
        """
        Extract the keypoints for each body part (arm, leg, torso).
        """
        arm_data = self._extract_and_swap(keypoint, self.body[0])
        leg_data = self._extract_and_swap(keypoint, self.body[1])
        torso_data = self._extract_and_swap(keypoint, self.body[2])
        return {"arm_data": arm_data, "leg_data": leg_data, "torso_data": torso_data}

    def set_pressure_map_data(self, pressure_map_data):
        """
        Set pressure map data for visualization.
        """
        row, col = pressure_map_data.shape[1:3]  # Get the row and column size of the pressure map
        self.mat_x = np.linspace(-17457, 17741, col)  # Create x-axis values for pressure map
        self.mat_y = np.linspace(5051, -5051, row)  # Create y-axis values for pressure map
        self.pressure_map_data = pressure_map_data  # Store pressure map data

    def set_mat_corner_data(self, keypoint):
        """
        Set data for mat corner (optional body part).
        """
        self.mat_tracker_data = self._extract_and_swap(keypoint, self.body[3])

    def set_frames(self):
        """
        Create frames for each time step. Each frame contains the 3D traces for arm, leg, and torso.
        """
        self.fig = go.Figure(
            frames=[go.Frame(
                data=[go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(size=0))],
                name=str(k)) for k in range(self.idx)]
        )

    def add_body_frames(self, body_data, color=['black', 'red', 'blue']):
        """
        Add frames for body data (arm, leg, torso) for each time step.
        """
        arm_data = body_data["arm_data"]
        leg_data = body_data["leg_data"]
        torso_data = body_data["torso_data"]

        for k, frame in enumerate(self.fig.frames):
            frame.data = list(frame.data) + [
                self._create_trace(arm_data, k, color=color[0]),
                self._create_trace(leg_data, k, color=color[1]),
                self._create_trace(torso_data, k, color=color[2])
            ]


    def add_mat_corner_frames(self):
        """
        Add frames for mat corner data, if it exists.
        """
        for k, frame in enumerate(self.fig.frames):
            frame.data = list(frame.data) + [
                self._create_trace(self.mat_tracker_data, k, color='green')]

        self.fig.add_trace(self._create_trace(self.mat_tracker_data, 0, color='green'))

    def add_pressure_map_frames(self):
        """
        Add frames for the pressure map data.
        """
        for k, frame in enumerate(self.fig.frames):
            frame.data = list(frame.data) + [
                go.Surface(x=self.mat_x, y=self.mat_y, z=self.pressure_map_data[k])]

    def save(self, file_path):
        """
        Save the figure as an HTML file.
        """
        self.fig.write_html(file_path)

    def _extract_and_swap(self, keypoint, indices):
        """
        Extract and swap axes for body data to match the plotting format.
        """
        if indices is None:
            return None
        data = np.swapaxes([keypoint[:, i - 1, :] for i in indices], 0, 1)
        return np.swapaxes(data, 1, 2)[:self.idx]

    def _create_trace(self, data, k, color):
        """
        Create a 3D scatter trace for a body part.
        """
        if data is None:
            return go.Scatter3d(x=[], y=[], z=[])
        return go.Scatter3d(
            x=data[k, 0, :], y=data[k, 1, :], z=data[k, 2, :],
            marker=dict(size=4, color=color),
            line=dict(width=5, color=color)
        )

    def _add_initial_traces(self, body_data, color):
        """
        Add the initial frame data (for frame 0).
        """
        arm_data = body_data["arm_data"]
        leg_data = body_data["leg_data"]
        torso_data = body_data["torso_data"]
        self.fig.add_trace(self._create_trace(arm_data, 0, color=color[0]))
        self.fig.add_trace(self._create_trace(leg_data, 0, color=color[1]))
        self.fig.add_trace(self._create_trace(torso_data, 0, color=color[2]))

    def frame_args(self, duration):
        """
        Return the animation frame arguments.
        """
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    def plot(self):
        """
        Create and show the plot, including sliders and animation controls.
        """
        sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], self.frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(self.fig.frames)
                ],
            }
        ]

        # Update layout with axis ranges and plot settings
        self.fig.update_layout(
            title='Body Visualisation',
            width=600,
            height=500,
            scene=dict(
                aspectmode='cube',  # Ensures equal scaling across all axes
                xaxis=dict(
                    title='X',  # Label for X axis
                    range=[self.axis["x_min"], self.axis["x_max"]],  # Set range based on min and max of X axis
                ),
                yaxis=dict(
                    title='Y',  # Label for Y axis
                    range=[self.axis["y_min"], self.axis["y_max"]],  # Set range based on min and max of Y axis
                ),
                zaxis=dict(
                    title='Z',  # Label for Z axis
                    range=[self.axis["z_min"], self.axis["z_max"]],  # Set range based on min and max of Z axis
                ),
            ),
            updatemenus=[{
                "buttons": [
                    {"args": [None, self.frame_args(50)], "label": "&#9654;", "method": "animate"},
                    {"args": [[None], self.frame_args(50)], "label": "&#9724;", "method": "animate"},
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }],
            sliders=sliders,
            showlegend=False,
        )
        self.fig.show()
