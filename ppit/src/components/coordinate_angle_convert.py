import collections

import numpy as np
from ppit.src.utils import timing
from ppit.src.exception import CustomException
from ppit.src.logger import logging

DEFAULT_SKELETON = {'hips': [[-111, 9557, 15726]],
                     'lefthip': [[4118,	-9089,	15819]],
                     'righthip': [[-4340,	-10026,	15634]],
                     'righttoe': [-3029,	-11039,	149],
                     'lefttoe': [2665,	-10929,	81],
                     'neck': [-86, -8889, 28144],
                     'leftshoulder': [[3493,	-8603,	28180]],
                     'leftwrist': [4527,	-10497,	15943],
                     'rightshoulder': [[-3665,-9175,28108]],
                     'rightwrist':[[-4613,	-10789,	16114]],
                     'leftelbow': [[4118,	-9185,	21457]],
                    'rightelbow': [-5253,	-9424,	21924],
                    'rightknee': [[-2358,	-8777,	8458]],
                    'leftknee':[[1997,	-8607,	8724]],
                    'rightfoot':[[-2491,	-7293,	770]],
                    'leftfoot':[[2268,	-7095,	686]]
                    }

class Converter():

    def __init__(self, joint_to_index:dict):
        """
        Initializes the Converter with a mapping from joints to indices.

        :param joint_to_index: Dictionary mapping joints to their corresponding indices.
        """
        self.joint_to_index = joint_to_index
        self.index_to_joint = {idx: joint for idx, joint in joint_to_index}

        self.kpts = {}
        self.kpts['joints'] = list(self.joint_to_index.keys())
        hierarchy = {'hips': [],
                     'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftfoot': ['leftknee', 'lefthip', 'hips'],
                     'righthip': ['hips'], 'rightknee': ['righthip', 'hips'],
                     'rightfoot': ['rightknee', 'righthip', 'hips'],
                     'righttoe': ['rightfoot', 'rightknee', 'righthip', 'hips'],
                     'lefttoe': ['leftfoot', 'leftknee', 'lefthip', 'hips'],
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
        for key, k_index in self.joint_to_index.items():
            self.kpts[key] = coordinates_array[:, k_index]
        ##############################


        self.add_hips_and_neck(self.kpts)
        self.median_filter()

        # Normalise the skeleton, optional
        # self.get_bone_lengths()
        # self.get_base_skeleton(filtered_kpts)
        

        angles_dict = self.calculate_joint_angles()

        ######################Turn dict to array
        angles_array = [angles_dict[self.index_to_joint[idx] + '_angles'] for idx in sorted(self.joint_to_index.values())]


        return np.swapaxes(angles_array, 0, 1)



    def angle2coordinate(self, angles_array: np.ndarray, root_position:np.ndarray, root_rotation:np.ndarray) -> np.ndarray:
        assert angles_array.shape[-1] == 3, ("Angle array should have format in (x,y,z) axis, data shape "
                                                   "should be (frames, joint number, 3)")


        coordinates_dict = collections.defaultdict(list)
        for framenum in range(self.kpts['hips'].shape[0]):

            # get a dictionary containing the rotations for the current frame
            frame_rotations = {}
            html_data_frame = []
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

    def get_bone_lengths(self, default=True):

        """
        We have to define an initial skeleton pose(T pose).
        In this case we need to known the length of each bone.
        Here we calculate the length of each bone from data
        """

        bone_lengths = {}
        for joint in self.kpts['joints']:
            if joint == 'hips': continue
            parent = self.kpts['hierarchy'][joint][0]

            if default:
                joint_kpts = DEFAULT_SKELETON[joint]
                parent_kpts = DEFAULT_SKELETON[parent]
            else:
                joint_kpts = self.kpts[joint]
                parent_kpts = self.kpts[parent]

            _bone = joint_kpts - parent_kpts
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
        offset_directions['leftfoot'] = np.array([0, -1, 0])
        offset_directions['lefttoe'] = np.array([0, -1, 0])

        offset_directions['righthip'] = np.array([1, 0, 0])
        offset_directions['rightknee'] = np.array([0, -1, 0])
        offset_directions['rightfoot'] = np.array([0, -1, 0])
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
        _set_length('foot')
        _set_length('toe')
        _set_length('shoulder')
        _set_length('elbow')
        _set_length('wrist')
        base_skeleton['neck'] = offset_directions['neck'] * (body_lengths['neck'] / normalization)

        self.kpts['offset_directions'] = offset_directions
        self.kpts['base_skeleton'] = base_skeleton
        self.kpts['normalization'] = normalization
        return

    def add_hips_and_neck(self):
        # we add two new keypoints which are the mid point between the hips and mid point between the shoulders

        # add hips kpts
        hips = (self.kpts['lefthip'] + self.kpts['righthip']) / 2
        self.kpts['hips'] = hips
        self.kpts['joints'].append('hips')

        # add neck self.kpts
        neck = (self.kpts['leftshoulder'] + self.kpts['rightshoulder']) / 2
        self.kpts['neck'] = neck
        self.kpts['joints'].append('neck')
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
        thetaz, thetay, thetax = utils.Decompose_R_ZYX(C)

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
