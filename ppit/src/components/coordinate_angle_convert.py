import collections

import numpy as np
from ppit.src.utils import timing
from ppit.src.exception import CustomException
from ppit.src.logger import logging
from ppit.src.utils import reader, writer

DEFAULT_SKELETON = {'hips': [-111, 9557, 15726],
                     'lefthip': [4118,	-9089,	15819],
                     'righthip': [-4340,	-10026,	15634],
                     'righttoe': [-3029,	-11039,	149],
                     'lefttoe': [2665,	-10929,	81],
                     'neck': [-86, -8889, 28144],
                     'leftshoulder': [3493,	-8603,	28180],
                     'leftwrist': [4527,	-10497,	15943],
                     'rightshoulder': [-3665,-9175,28108],
                     'rightwrist':[-4613,	-10789,	16114],
                     'leftelbow': [4118,	-9185,	21457],
                    'rightelbow': [-5253,	-9424,	21924],
                    'rightknee': [-2358,	-8777,	8458],
                    'leftknee':[1997,	-8607,	8724],
                    'rightankle':[-2491,	-7293,	770],
                    'leftankle':[2268,	-7095,	686]
                    }

class Converter():

    def __init__(self, config_path:str=None):
        """
        Initializes the Converter with a mapping from joints to indices.

        :param joint_to_index: Dictionary mapping joints to their corresponding indices.
        In the keypoint coordinates data file, joint index does not have neck and hips
        In the keypoint rotatioin angle data file, joint index need to append 'hips' and 'neck' as the last two index
        """
        # Load configuration
        config = reader(config_path)
        self.joint_to_index = {joint_name.lower().replace(" ", ""): joint_index for joint_name, joint_index in
                       config["joints"].items()}

        self.index_to_joint = {idx: joint for joint, idx in self.joint_to_index.items()}

        self.kpts = {}
        self.kpts['joints'] = list(self.joint_to_index.keys())

        self.kpts['joints']+=["hips", "neck"]
        hierarchy = {'hips': [],
                     'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftankle': ['leftknee', 'lefthip', 'hips'],
                     'righthip': ['hips'], 'rightknee': ['righthip', 'hips'],
                     'rightankle': ['rightknee', 'righthip', 'hips'],
                     'righttoe': ['rightankle', 'rightknee', 'righthip', 'hips'],
                     'lefttoe': ['leftankle', 'leftknee', 'lefthip', 'hips'],
                     'neck': ['hips'],
                     'leftshoulder': ['neck', 'hips'], 'leftelbow': ['leftshoulder', 'neck', 'hips'],
                     'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
                     'rightshoulder': ['neck', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'hips'],
                     'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'hips']
                     }

        self.kpts['hierarchy'] = hierarchy
        self.kpts['root_joint'] = 'hips'
        self.get_bone_lengths(default=True)
        self.get_base_skeleton()
        logging.info(f"Finished initialising keypoint angle converter component")

        return



    @timing
    def coordinate2angle(self, coordinates_array: np.ndarray) -> np.ndarray:
        """
        Converts a coordinates array (x, y, z) into an array of angles.

        Currently, this is a placeholder function that returns the input array unchanged.

        :param coordinates_array: A numpy array of shape (frames, joint_number, 3) representing coordinates.
        :return: A numpy array of the same shape as input, currently unchanged.
        """
        assert coordinates_array.shape[-1] == 3, ("Coordinates array should have format in (x,y,z) axis, data shape "
                                                   "should be (frames, joint number, 3)")


        #################### Turn array to dict
        for joint, idx in self.joint_to_index.items():
            self.kpts[joint] = coordinates_array[:, idx-1]
        ##############################


        self.add_hips_and_neck()
        self.median_filter()

        # Normalise the skeleton, optional
        # self.get_bone_lengths()
        # self.get_base_skeleton(filtered_kpts)
        

        self.calculate_joint_angles()

        ######################Turn dict to array
        angles_array = [self.kpts[self.index_to_joint[idx] + '_angles'] for idx in sorted(self.joint_to_index.values())]
        angles_array.append(self.kpts['hips_angles'])
        angles_array.append(self.kpts['neck_angles'])

        return np.swapaxes(angles_array, 0, 1)


    @timing
    def angle2coordinate(self, angles_array: np.ndarray) -> np.ndarray:
        assert angles_array.shape[-1] == 3, ("Angle array should have format in (x,y,z) axis, data shape "
                                                   "should be (frames, joint number, 3)")

        # Assign angle array data to dictionary
        for idx, joint in self.index_to_joint.items():
            self.kpts[joint + '_angles'] = angles_array[:, idx-1, :]

        # Add hips and neck data from the last two columns
        self.kpts['hips_angles'] = angles_array[:, idx, :]
        self.kpts['neck_angles'] = angles_array[:, idx+1, :]


        coordinates_dict = collections.defaultdict(list)

        self.kpts['hips'] = np.ones(self.kpts['leftankle_angles'].shape) # change to hips rotation angle later  TODO
        self.kpts['joints'] = list(self.index_to_joint.values())
        if "hips" not in self.kpts['joints']: self.kpts['joints'] += ['hips']
        if "neck" not in self.kpts['joints']: self.kpts['joints'] += ['neck']


        for framenum in range(self.kpts['hips'].shape[0]):

            # get a dictionary containing the rotations for the current frame
            frame_rotations = {}

            for joint in self.kpts['joints']:
                frame_rotations[joint] = self.kpts[joint + '_angles'][framenum]

            # for plotting
            for _j in self.kpts['joints']:
                if _j == 'hips': continue

                # get hierarchy of how the joint connects back to root joint
                hierarchy = self.kpts['hierarchy'][_j]

                # get the current position of the parent joint
                r1 = self.kpts['hips'][framenum] / self.kpts['normalization']   # This can be substituted with [0, 0, 1]
                for parent in hierarchy:
                    if parent == 'hips': continue
                    R = self.get_rotation_chain(parent, self.kpts['hierarchy'][parent], frame_rotations)
                    # get rotation chain (uplevel connecting joint, all parent of uplevel connectiong joint, joint angles)

                    r1 = r1 + R @ self.kpts['base_skeleton'][parent]

                # up to now, we are calculating the position of the up level joint
                # get the current position of the joint. Note: r2 is the final position of the joint. r1 is simply calculated for plotting.
                # get rotation chain (first connecting joint, all parent, joint angles)
                r2 = r1 + self.get_rotation_chain(hierarchy[0], hierarchy, frame_rotations) @ self.kpts['base_skeleton'][_j]
                coordinates_dict[_j].append(r2)

        coordinates_array = [coordinates_dict[self.index_to_joint[idx]] for idx in
                        sorted(self.joint_to_index.values())]
        return np.swapaxes(coordinates_array, 0, 1)

    def get_rotation_chain(self, joint, hierarchy, frame_rotations):

        hierarchy = hierarchy[::-1]

        # this code assumes ZXY rotation order
        R = np.eye(3)
        for parent in hierarchy:
            angles = frame_rotations[parent]
            _R = get_R_z(angles[0]) @ get_R_x(angles[1]) @ get_R_y(angles[2])
            R = R @ _R

        return R
    def median_filter(self, window_size=3):

        from scipy.signal import medfilt

        # apply median filter to get rid of poor keypoints estimations
        for joint in self.kpts['joints']:
            joint_kpts = self.kpts[joint]
            xs = joint_kpts[:, 0]
            ys = joint_kpts[:, 1]
            zs = joint_kpts[:, 2]
            xs = medfilt(xs, window_size)
            ys = medfilt(ys, window_size)
            zs = medfilt(zs, window_size)
            self.kpts[joint] = np.stack([xs, ys, zs], axis=-1)

        return

    def get_joint_rotations(self, joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos):

        _invR = np.eye(3)
        for i, parent_name in enumerate(joints_hierarchy[joint_name]):
            if i == 0: continue
            _r_angles = frame_rotations[parent_name]
            R = get_R_z(_r_angles[0]) @ get_R_x(_r_angles[1]) @ get_R_y(_r_angles[2])
            _invR = _invR @ R.T
        b = _invR @ (frame_pos[joint_name] - frame_pos[joints_hierarchy[joint_name][0]])

        _R = Get_R2(joints_offsets[joint_name], b)
        tz, ty, tx = Decompose_R_ZXY(_R)
        joint_rs = np.array([tz, tx, ty])
        return joint_rs

    def get_bone_lengths(self, default=True):

        """
        We have to define an initial skeleton pose(T pose).
        In this case we need to know the length of each bone.
        Here we calculate the length of each bone from data
        """

        bone_lengths = {}
        for joint in DEFAULT_SKELETON:
            if joint == 'hips': continue
            parent = self.kpts['hierarchy'][joint][0]

            if default:
                joint_kpts = DEFAULT_SKELETON[joint]
                parent_kpts = DEFAULT_SKELETON[parent]
            else:
                joint_kpts = self.kpts[joint]
                parent_kpts = self.kpts[parent]

            _bone = np.subtract(joint_kpts, parent_kpts)
            _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis=-1))

            _bone_length = np.median(_bone_lengths)
            bone_lengths[joint] = _bone_length

        self.kpts['bone_lengths'] = bone_lengths
        return self.kpts

    def get_base_skeleton(self, normalization_bone='leftknee'):

        # this defines a generic skeleton to which we can apply rotations to
        body_lengths = self.kpts['bone_lengths']

        # define skeleton offset directions
        offset_directions = {}
        offset_directions['lefthip'] = np.array([-1, 0, 0])
        offset_directions['leftknee'] = np.array([0, -1, 0])
        offset_directions['leftankle'] = np.array([0, -1, 0])
        offset_directions['lefttoe'] = np.array([0, -1, 0])

        offset_directions['righthip'] = np.array([1, 0, 0])
        offset_directions['rightknee'] = np.array([0, -1, 0])
        offset_directions['rightankle'] = np.array([0, -1, 0])
        offset_directions['righttoe'] = np.array([0, -1, 0])

        offset_directions['neck'] = np.array([0, 1, 0])

        offset_directions['leftshoulder'] = np.array([1, 0, 0])
        offset_directions['leftelbow'] = np.array([1, 0, 0])
        offset_directions['leftwrist'] = np.array([1, 0, 0])

        offset_directions['rightshoulder'] = np.array([-1, 0, 0])
        offset_directions['rightelbow'] = np.array([-1, 0, 0])
        offset_directions['rightwrist'] = np.array([-1, 0, 0])

        # set bone normalization length. Set to 1 if you dont want normalization
        normalization = self.kpts['bone_lengths'][normalization_bone]

        # base skeleton set by multiplying offset directions by measured bone lengths. In this case we use the average of two sided limbs. E.g left and right hip averaged
        base_skeleton = {'hips': np.array([0, 0, 0])}

        def _set_length(joint_type):
            base_skeleton['left' + joint_type] = offset_directions['left' + joint_type] * (
                        (body_lengths['left' + joint_type] + body_lengths['right' + joint_type]) / (2 * normalization))
            base_skeleton['right' + joint_type] = offset_directions['right' + joint_type] * (
                        (body_lengths['left' + joint_type] + body_lengths['right' + joint_type]) / (2 * normalization))

        _set_length('hip')
        _set_length('knee')
        _set_length('ankle')
        _set_length('toe')
        _set_length('shoulder')
        _set_length('elbow')
        _set_length('wrist')
        base_skeleton['neck'] = offset_directions['neck'] * (body_lengths['neck'] / normalization)

        self.kpts['offset_directions'] = offset_directions
        self.kpts['base_skeleton'] = {'hips': np.array([0, 0, 0]),
                                     'lefthip': np.array([-0.5,  0,  0]),
                                     'righthip': np.array([0.5, 0, 0]),
                                     'leftknee': np.array([0., -1,  0.]),
                                     'rightknee': np.array([0, -1,  0]),
                                     'leftankle': np.array([0, -1,  0]),
                                     'rightankle': np.array([ 0, -1,  0]),
                                     'lefttoe': np.array([ 0, -0.5,  0]),
                                     'righttoe': np.array([ 0, -0.5,  0]),
                                     'leftshoulder': np.array([0.5, 0, 0]),
                                     'rightshoulder': np.array([-0.5,  0,  0]),
                                     'leftelbow': np.array([0.9, 0, 0]),
                                     'rightelbow': np.array([-0.9,  0,  0]),
                                     'leftwrist': np.array([0.8, 0, 0]),
                                     'rightwrist': np.array([-0.8,  0,  0]),
                                     'neck': np.array([0, 1.3, 0])}
        self.kpts['normalization'] = normalization
        return

    def add_hips_and_neck(self):
        # we add two new keypoints which are the mid point between the hips and mid point between the shoulders

        if 'lefthip' in self.kpts.keys() and 'righthip' in self.kpts.keys():
            # add hips kpts
            hips = (self.kpts['lefthip'] + self.kpts['righthip']) / 2
            self.kpts['hips'] = hips
        if 'leftshoulder' in self.kpts.keys() and 'rightshoulder' in self.kpts.keys():
            # add neck self.kpts
            neck = (self.kpts['leftshoulder'] + self.kpts['rightshoulder']) / 2
            self.kpts['neck'] = neck
            # define the hierarchy of the joints
        return

    # calculate the rotation of the root joint with respect to the world coordinates
    def get_hips_position_and_rotation(self, frame_pos, root_joint='hips', root_define_joints=['righthip', 'neck']):

        # root position is saved directly
        root_position = frame_pos[root_joint]

        # calculate unit vectors of root joint
        root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
        root_u = root_u / np.sqrt(np.sum(np.square(root_u)))
        root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
        root_v = root_v / np.sqrt(np.sum(np.square(root_v)))
        root_w = np.cross(root_u, root_v)

        # Make the rotation matrix
        C = np.array([root_u, root_v, root_w]).T
        thetaz, thetay, thetax = Decompose_R_ZYX(C)

        root_rotation = np.array([thetaz, thetax, thetay])

        return root_position, root_rotation


    def calculate_joint_angles(self):

        # set up emtpy container for joint angles
        for joint in self.kpts['joints']:
            self.kpts[joint + '_angles'] = []

        for framenum in range(self.kpts['hips'].shape[0]):

            # get the keypoints positions in the current frame
            frame_pos = {}

            for joint in self.kpts['joints']:
                frame_pos[joint] = self.kpts[joint][framenum]
            root_position, root_rotation = self.get_hips_position_and_rotation(frame_pos)
            frame_rotations = {'hips': root_rotation}

            # center the body pose
            for joint in self.kpts['joints']:
                frame_pos[joint] = frame_pos[joint] - root_position

            # get the max joints connectsion
            max_connected_joints = 0
            for joint in self.kpts['joints']:
                if len(self.kpts['hierarchy'][joint]) > max_connected_joints:
                    max_connected_joints = len(self.kpts['hierarchy'][joint])

            depth = 2
            while (depth <= max_connected_joints):
                for joint in self.kpts['joints']:

                    if len(self.kpts['hierarchy'][joint]) == depth:
                        joint_rs = self.get_joint_rotations(joint, self.kpts['hierarchy'], self.kpts['offset_directions'],
                                                       frame_rotations, frame_pos)
                        parent = self.kpts['hierarchy'][joint][0]
                        frame_rotations[parent] = joint_rs
                depth += 1

            # for completeness, add zero rotation angles for endpoints. This is not necessary as they are never used.
            for _j in self.kpts['joints']:
                if _j not in list(frame_rotations.keys()):
                    frame_rotations[_j] = np.array([0., 0., 0.])

            # update dictionary with current angles.
            for joint in self.kpts['joints']:
                self.kpts[joint + '_angles'].append(frame_rotations[joint])

        # convert joint angles list to numpy arrays.
        for joint in self.kpts['joints']:
            self.kpts[joint + '_angles'] = np.array(self.kpts[joint + '_angles'])

        return


#general rotation matrices
def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R


#calculate rotation matrix to take A vector to B vector
def Get_R(A,B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    #get products
    dotprod = np.sum(uA * uB)
    crossprod = np.sqrt(np.sum(np.square(np.cross(uA,uB)))) #magnitude

    #get new unit vectors
    u = uA
    v = uB - dotprod*uA
    v = v/np.sqrt(np.sum(np.square(v)))
    w = np.cross(uA, uB)
    w = w/np.sqrt(np.sum(np.square(w)))

    #get change of basis matrix
    C = np.array([u, v, w])

    #get rotation matrix in new basis
    R_uvw = np.array([[dotprod, -crossprod, 0],
                      [crossprod, dotprod, 0],
                      [0, 0, 1]])

    #full rotation matrix
    R = C.T @ R_uvw @ C
    return R

#Same calculation as above using a different formalism
def Get_R2(A, B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    v = np.cross(uA, uB)
    s = np.sqrt(np.sum(np.square(v)))
    c = np.sum(uA * uB)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx@vx*((1-c)/s**2)

    return R


#decomposes given R matrix into rotation along each axis. In this case Rz @ Ry @ Rx
def Decompose_R_ZYX(R):

    #decomposes as RzRyRx. Note the order: ZYX <- rotation by x first
    thetaz = np.arctan2(R[1,0], R[0,0])
    thetay = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    thetax = np.arctan2(R[2,1], R[2,2])

    return thetaz, thetay, thetax

def Decompose_R_ZXY(R):
    #decomposes as RzRXRy. Note the order: ZXY <- rotation by y first

    thetaz = np.arctan2(-R[0,1], R[1,1])
    thetay = np.arctan2(-R[2,0], R[2,2])
    thetax = np.arctan2(R[2,1], np.sqrt(R[2,0]**2 + R[2,2]**2))
    return thetaz, thetay, thetax

if __name__=="__main__":



    keypoint_data = reader("../../../data/20230626_5_set_3_1.p")['keypoint']

    joint_coords_converter1 = Converter("../../../config/data_processor_example.yml")
    angle_data = joint_coords_converter1.coordinate2angle(keypoint_data)

    joint_coords_converter2 = Converter("../../../config/data_processor_example.yml")
    converted_data = joint_coords_converter2.angle2coordinate(angle_data)

    # converted_data[:, :, [0, 1, 2]] = converted_data[:, :, [0, 2, 1]]
    # converted_data += np.array([0, 0, 2])
    converted_data *= 5000

    body = [[6, 4, 2, 1, 3, 5],  # arm joints
            [12, 10, 8, 7, 9, 11],  # leg joints
            [1, 2, 8, 7, 1],  # torso joints
            [15, 13, 14],  # Mat 1 , 2, 3
            ]

    arm_data = np.swapaxes([keypoint_data[:, i - 1, :] for i in body[0]], 0, 1).astype(int)
    arm_data = np.swapaxes(arm_data, 1, 2).astype(int)
    leg_data = np.swapaxes([keypoint_data[:, i - 1, :] for i in body[1]], 0, 1).astype(int)
    leg_data = np.swapaxes(leg_data, 1, 2).astype(int)
    torso_data = np.swapaxes([keypoint_data[:, i - 1, :] for i in body[2]], 0, 1).astype(int)
    torso_data = np.swapaxes(torso_data, 1, 2).astype(int)

    converted_arm_data = np.swapaxes([converted_data[:, i - 1, :] for i in body[0]], 0, 1).astype(int)
    converted_arm_data = np.swapaxes(converted_arm_data, 1, 2).astype(int)
    converted_leg_data = np.swapaxes([converted_data[:, i - 1, :] for i in body[1]], 0, 1).astype(int)
    converted_leg_data = np.swapaxes(converted_leg_data, 1, 2).astype(int)
    converted_torso_data = np.swapaxes([converted_data[:, i - 1, :] for i in body[2]], 0, 1).astype(int)
    converted_torso_data = np.swapaxes(converted_torso_data, 1, 2).astype(int)

    # Define frames
    import plotly.graph_objects as go

    fig = go.Figure(frames=[go.Frame(
        data=[

            go.Scatter3d(
                x=arm_data[k, 0, :],
                y=arm_data[k, 1, :],
                z=arm_data[k, 2, :],
                marker=dict(
                    size=0,
                    color='black',
                    #                                                                     colorscale='Inferno',
                ),
                line=dict(
                    width=5,
                    color='black',
                    #                                                                     colorscale='Inferno'
                )
            ),
            go.Scatter3d(  # joint marker
                x=leg_data[k, 0, :],
                y=leg_data[k, 1, :],
                z=leg_data[k, 2, :],
                marker=dict(
                    size=0,
                    color='red',
                    #                                                                     colorscale='Inferno',
                ),
                line=dict(
                    width=5,
                    color='red',
                    #                                                                     colorscale='Inferno'
                )
            ),
            go.Scatter3d(  # joint marker
                x=torso_data[k, 0, :],
                y=torso_data[k, 1, :],
                z=torso_data[k, 2, :],
                marker=dict(
                    size=0,
                    color='blue',
                    #                                                                     colorscale='Inferno',
                ),
                line=dict(
                    width=5,
                    color='blue',
                    #                                                                     colorscale='Inferno'
                )
            ),

            go.Scatter3d(
                x=converted_arm_data[k, 0, :],
                y=converted_arm_data[k, 1, :],
                z=converted_arm_data[k, 2, :],
                marker=dict(
                    size=0,
                    color='black',
                    #                                                                     colorscale='Inferno',
                ),
                line=dict(
                    width=5,
                    color='black',
                    #                                                                     colorscale='Inferno'
                )
            ),
            go.Scatter3d(  # joint marker
                x=converted_leg_data[k, 0, :],
                y=converted_leg_data[k, 1, :],
                z=converted_leg_data[k, 2, :],
                marker=dict(
                    size=0,
                    color='red',
                    #                                                                     colorscale='Inferno',
                ),
                line=dict(
                    width=5,
                    color='red',
                    #                                                                     colorscale='Inferno'
                )
            ),
            go.Scatter3d(  # joint marker
                x=converted_torso_data[k, 0, :],
                y=converted_torso_data[k, 1, :],
                z=converted_torso_data[k, 2, :],
                marker=dict(
                    size=0,
                    color='blue',
                    #                                                                     colorscale='Inferno',
                ),
                line=dict(
                    width=5,
                    color='blue',
                    #                                                                     colorscale='Inferno'
                )
            ),

        ]
        ,
        name=str(k)  # you need to name the frame for the animation to behave properly
    )
        for k in range(len(keypoint_data))])

    fig.add_trace(go.Scatter3d(
        x=arm_data[0, 0, :],
        y=arm_data[0, 1, :],
        z=arm_data[0, 2, :],
        marker=dict(
            size=0,
            color='black',
            #                                                                     colorscale='Inferno',
        ),
        line=dict(
            width=5,
            color='black',
            #                                                                     colorscale='Inferno'
        )
    ),
    )

    fig.add_trace(go.Scatter3d(  # joint marker
        x=leg_data[0, 0, :],
        y=leg_data[0, 1, :],
        z=leg_data[0, 2, :],
        marker=dict(
            size=0,
            color='red',
            #                                                                     colorscale='Inferno',
        ),
        line=dict(
            width=5,
            color='red',
            #                                                                     colorscale='Inferno'
        )
    ),
    )
    fig.add_trace(go.Scatter3d(  # joint marker
        x=torso_data[0, 0, :],
        y=torso_data[0, 1, :],
        z=torso_data[0, 2, :],
        marker=dict(
            size=0,
            color='blue',
            #                                                                     colorscale='Inferno',
        ),
        line=dict(
            width=5,
            color='blue',
            #                                                                     colorscale='Inferno'
        )
    ),
    )

    fig.add_trace(go.Scatter3d(
        x=converted_arm_data[0, 0, :],
        y=converted_arm_data[0, 1, :],
        z=converted_arm_data[0, 2, :],
        marker=dict(
            size=0,
            color='black',
            #                                                                     colorscale='Inferno',
        ),
        line=dict(
            width=5,
            color='black',
            #                                                                     colorscale='Inferno'
        )
    ),
    )

    fig.add_trace(go.Scatter3d(  # joint marker
        x=converted_leg_data[0, 0, :],
        y=converted_leg_data[0, 1, :],
        z=converted_leg_data[0, 2, :],
        marker=dict(
            size=0,
            color='red',
            #                                                                     colorscale='Inferno',
        ),
        line=dict(
            width=5,
            color='red',
            #                                                                     colorscale='Inferno'
        )
    ),
    )
    fig.add_trace(go.Scatter3d(  # joint marker
        x=converted_torso_data[0, 0, :],
        y=converted_torso_data[0, 1, :],
        z=converted_torso_data[0, 2, :],
        marker=dict(
            size=0,
            color='blue',
            #                                                                     colorscale='Inferno',
        ),
        line=dict(
            width=5,
            color='blue',
            #                                                                     colorscale='Inferno'
        )
    ),
    )


    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }


    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title='body movement',
        width=600,
        height=500,
        #         paper_bgcolor='black',
        #         plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            zaxis=dict(
                range=[-32767, 32767],
                autorange=False,
                # #                                color='white',
                #                               ticksuffix = "%"
            ),
            aspectratio=dict(x=2, y=2, z=1),
            xaxis=dict(
                range=[-32767, 32767],
                autorange=False,
                #                         showgrid=False,
                #                               showticklabels=False,
            ),

            #
            yaxis=dict(
                range=[-32767, 32767],
                autorange=False,
                #                             showgrid=False,
                #                               showticklabels=False,
            ),
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
        showlegend=False,
    )

    fig.show()
    fig.write_html("test.html")



