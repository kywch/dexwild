import cv2
import threading
import os
import time
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R
from dexwild_utils.pose_utils import pose7d_to_mat, mat_to_pose7d, pose6d_to_mat
import pickle

def get_mask(image_points, width, height):
    
    output_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # make a mask of the cube
    # Compute the  convex hull of the 8 projected points
    # (ensures a valid polygon even if points are not in order)
    if image_points is None:
        return output_image
    
    hull = cv2.convexHull(image_points)
    
    #plot the points
    for point in image_points:
        cv2.circle(output_image, tuple(point), 5, (0, 0, 255), -1)
    
    # Fill the hull region in the output mask with black (0)
    # If output_image is single-channel: fill color = 0
    # If output_image is 3-channel: fill color = (0,0,0)
    if output_image.ndim == 2:
        cv2.fillConvexPoly(output_image, hull, 0)
    else:
        cv2.fillConvexPoly(output_image, hull, (0, 0, 0))
    return output_image

class CubeTracker:
    def __init__(self, camera_matrix, dist_coeffs, cube_size, marker_size, marker_ids, face_corners_3d, transformation):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cube_size = cube_size
        self.half_c = cube_size / 2.0
        self.marker_size = marker_size
        self.half_m = marker_size / 2.0
        self.marker_ids = marker_ids
        self.face_corners_3d = face_corners_3d
        self.transformation = transformation
        
        self.cm_to_m = 0.01
        self.m_to_cm = 100.0
        
        assert len(self.marker_ids) == 6
        
        self.marker_orientations = {
            # marker_ids[0]: '+Z', # don't need this one
            marker_ids[1]: '-Z',
            marker_ids[2]: '+Y',
            marker_ids[3]: '-Y',
            marker_ids[4]: '+X',
            marker_ids[5]: '-X'
        }
        
        # print("Marker orientations:", self.marker_orientations)

        # Build a dictionary: markerId -> 4 corners in 3D
        self.marker_id_to_3d_corners = {}
        for m_id in self.marker_ids:
            if m_id not in self.marker_orientations:
                continue
            orientation = self.marker_orientations[m_id]
            self.marker_id_to_3d_corners[m_id] = self.face_corners_3d(orientation, self.half_c, self.half_m)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        
        self.detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict)
    
    def set_intrinsics(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def estimate_pose(self, image, depth = None):
        # ASSUME BGR image!!
        
        # check if its gray already
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            # print("No markers detected.")
            return False, None, None
        
        # 3. Gather 2D–3D correspondences for all detected markers
        all_3d_points = []
        all_2d_points = []
        
        for (corners, marker_id) in zip(corners_list, ids):
            marker_id = int(marker_id[0]) # marker_id is a 1x1 array
            
            if marker_id not in self.marker_id_to_3d_corners:
                continue  # skip markers not in our dictionary

            corners_2d = corners.reshape(-1, 2)  # shape (4,2)
            corners_3d = self.marker_id_to_3d_corners[marker_id]

            all_3d_points.append(corners_3d)
            all_2d_points.append(corners_2d)

        if len(all_3d_points) == 0:
            # print("No cube faces (marker IDs) recognized in the image.")
            return False, None, None

        all_3d_points = np.concatenate(all_3d_points, axis=0)  # shape (4*num_markers, 3)
        all_2d_points = np.concatenate(all_2d_points, axis=0)  # shape (4*num_markers, 2)

        if depth is None: # NOTE USE RGB IMAGE Only
            return self.traditional_pnp(all_3d_points, all_2d_points)
        else: # NOTE USE DEPTH IMAGE
            success, rvec, tvec = self.depth_pnp(all_3d_points, all_2d_points, depth)
            if not success:
                # fallback to traditional PnP if depth fails
                return self.traditional_pnp(all_3d_points, all_2d_points)
            
            return success, rvec, tvec
        
    def traditional_pnp(self, all_3d_points, all_2d_points):
        # 4. Solve PnP to estimate the global pose of the cube
        
        success, rvec, tvec = cv2.solvePnP(
            all_3d_points,
            all_2d_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("solvePnP failed to find a consistent pose.")
            return False, None, None

        # Optional: refine the pose if multiple markers are visible
        if len(all_3d_points) >= 12:  # e.g. if 3 or more faces are detected
            rvec, tvec = cv2.solvePnPRefineLM(
                all_3d_points, all_2d_points, self.camera_matrix, self.dist_coeffs, rvec, tvec
            )

        return True, rvec, tvec
    
    def depth_pnp(self, all_3d_points, all_2d_points, depth):
        camera_pts_3d = []
        object_pts_3d = []

        
        for i in range(all_2d_points.shape[0]):
            u, v = all_2d_points[i]
            # known object coords
            Xo, Yo, Zo = all_3d_points[i]

            # get depth
            z_val = depth[int(v), int(u)]
            
            z_val = z_val * self.m_to_cm  # convert to cm 

            # skip invalid or negative depth
            if z_val <= 0.0 or np.isnan(z_val):
                # print("Invalid depth value:", z_val)
                continue

            # back-project into camera coords
            Xc = (u - self.camera_matrix[0, 2]) * z_val / self.camera_matrix[0, 0]
            Yc = (v - self.camera_matrix[1, 2]) * z_val / self.camera_matrix[1, 1]
            Zc = z_val

            camera_pts_3d.append([Xc, Yc, Zc])
            object_pts_3d.append([Xo, Yo, Zo])

        # Must have enough corners with valid depth
        if len(camera_pts_3d) < 4:
            return False, None, None

        camera_pts_3d = np.array(camera_pts_3d, dtype=np.float32)
        object_pts_3d = np.array(object_pts_3d, dtype=np.float32)

        # 3D->3D alignment via Horn's method (Kabsch)
        success_3d, R_c, t_c = self.rigid_transform_3D(object_pts_3d, camera_pts_3d)
        if not success_3d:
            # print("Failed to compute rigid transform.")
            return False, None, None

        # Convert R_c (3x3) to rvec
        rvec, _ = cv2.Rodrigues(R_c)
        tvec = t_c.reshape(3, 1)

        return True, rvec, tvec

    def get_pose(self, image, depth = None):

        success, rvec, tvec = self.estimate_pose(image, depth)

        if not success:
            return None
        
        position = tvec.flatten() * self.cm_to_m  # convert to meters
        rot_mat, _ = cv2.Rodrigues(rvec)
        quaternion = R.from_matrix(rot_mat).as_quat()
        
        pose_mat = pose7d_to_mat(np.concatenate([position, quaternion]))
        trans_mat = pose6d_to_mat(self.transformation)
        
        pose_mat = pose_mat @ trans_mat
        
        pose = mat_to_pose7d(pose_mat)
        
        return pose
    
    def get_mask_points(self, pose):

        if np.allclose(pose,  np.array([0., 0., 0., 0., 0., 0., 1.]), atol=1e-5):
            return None
        
        # undo transformation
        pose_mat = pose7d_to_mat(pose)
        trans_mat = pose6d_to_mat(self.transformation)
        
        pose_mat = pose_mat @ np.linalg.inv(trans_mat)
        
        pose = mat_to_pose7d(pose_mat)
    
        position = pose[:3] * self.m_to_cm
        rot_mat = R.from_quat(pose[3:]).as_matrix()
        
        # convert back to rvec
        tvec = position
        rvec, _ = cv2.Rodrigues(rot_mat)
        
        c = self.half_c   # e.g. 3.75
        
        cube_points = np.array([
            [-c, -c, -c],  # 0
            [ c, -c, -c],  # 1
            [ c,  c, -c],  # 2
            [-c,  c, -c],  # 3
            [-c, -c,  c],  # 4
            [ c, -c,  c],  # 5
            [ c,  c,  c],  # 6
            [-c,  c,  c],  # 7
        ], dtype=np.float32)

        image_points, _ = cv2.projectPoints(cube_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2).astype(int)
        
        return image_points
        
    def overlay_cube_pose(self, image, poses):
        output_image = image.copy()
        
        for key, pose in poses.items():
            if np.allclose(pose,  np.array([0., 0., 0., 0., 0., 0., 1.]), atol=1e-5):
                continue
            
            position = pose[:3] * self.m_to_cm
            rot_mat = R.from_quat(pose[3:]).as_matrix()
            
            # convert back to rvec
            tvec = position
            rvec, _ = cv2.Rodrigues(rot_mat)
            
            c = self.half_c   # e.g. 3.75
            m = self.half_m     # e.g. 2.25

            # --- 1) Define the 8 corners of a cube centered at (0,0,0) ---
            #     with faces at ±c in X, Y, Z
            cube_points = np.array([
                [-c, -c, -c],  # 0
                [ c, -c, -c],  # 1
                [ c,  c, -c],  # 2
                [-c,  c, -c],  # 3
                [-c, -c,  c],  # 4
                [ c, -c,  c],  # 5
                [ c,  c,  c],  # 6
                [-c,  c,  c],  # 7
            ], dtype=np.float32)

            # --- 2) Define marker vertices for just the 'top' and 'bottom' faces ---
            #     The bottom face is at Z = -c, the top face at Z = +c.
            #     Each marker is marker_size x marker_size, centered on that face.
            marker_faces = np.array([
                # Bottom face (z = -c)
                [-m, -m, -c],
                [ m, -m, -c],
                [ m,  m, -c],
                [-m,  m, -c],

                # Top face (z = +c)
                [-m, -m,  c],
                [ m, -m,  c],
                [ m,  m,  c],
                [-m,  m,  c]
            ], dtype=np.float32)

            # --- 3) Project the 3D cube corners and marker corners onto the 2D image plane ---
            
            image_points, _ = cv2.projectPoints(cube_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            marker_points, _ = cv2.projectPoints(marker_faces, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                
            image_points = image_points.reshape(-1, 2).astype(int)
            marker_points = marker_points.reshape(-1, 2).astype(int)

            # --- 4) Define the cube edges for drawing ---
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face edges
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face edges
                (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
            ]

            # --- 5) Draw the edges of the cube ---
            for start, end in edges:
                pt1 = tuple(image_points[start])
                pt2 = tuple(image_points[end])
                cv2.line(output_image, pt1, pt2, (0, 255, 0), 2)

            # --- 6) Draw the marker outlines on the 'top' and 'bottom' faces ---
            #     We'll draw them as quadrilaterals.
            for i in range(0, len(marker_points), 4):
                face_pts = marker_points[i:i+4]
                cv2.polylines(output_image, [face_pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # --- 7) Draw the coordinate axes for reference ---
            #     The "length" parameter sets how large the axis lines appear in the image.
            #     Using c or marker_size is typical. For clarity, let's use the full cube_size.
            axis_length = self.cube_size  # in the same cm units
        
            cv2.drawFrameAxes(output_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, axis_length)

        return output_image

    def rigid_transform_3D(self, A, B):
        """
        Compute the rigid transform (R, t) that aligns points A to B:
        B ~ R*A + t, with no scaling.

        A, B: (N, 3) arrays of corresponding 3D points
        Returns: (success, R(3x3), t(3, ))
        """
        if A.shape != B.shape or A.shape[0] < 3:
            return False, None, None

        # 1) Centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # 2) Subtract means
        AA = A - centroid_A
        BB = B - centroid_B

        # 3) Covariance + SVD
        H = AA.T @ BB  # 3x3
        U, S, Vt = np.linalg.svd(H)
        R_ = Vt.T @ U.T

        # 4) Correct reflection if needed
        if np.linalg.det(R_) < 0:
            Vt[2, :] *= -1
            R_ = Vt.T @ U.T

        # 5) Translation
        t_ = centroid_B - R_ @ centroid_A

        return True, R_, t_