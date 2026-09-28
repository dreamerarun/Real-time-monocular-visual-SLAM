"""
Real-time Visual SLAM with ORB-SLAM3 Style Features - v11 FIXED
- Fixed inverted coordinate system
- Fixed rotation controls in 3D viewer
- Persistent map points with localization
- Follow camera mode
- Save/Load map functionality
- Interactive 3D viewer with full controls
"""
import cv2
import numpy as np
import cuvslam
import time
import argparse
import pickle
import os
from collections import deque

# Try to import open3d for 3D visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not installed. 3D viewer disabled.")
    print("Install with: pip install open3d")


def rotation_matrix_from_quaternion(q):
    """Convert quaternion to rotation matrix"""
    # q = [x, y, z, w]
    x, y, z, w = q
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


class SLAM3DViewer:
    """ORB-SLAM3 style 3D visualization for SLAM map with FIXED coordinate system"""
    
    def __init__(self):
        if not HAS_OPEN3D:
            return
            
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('SLAM 3D Map Viewer', width=1024, height=768)
        
        # Create coordinate frame at origin
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        self.vis.add_geometry(self.coordinate_frame)
        
        # Point cloud for map points (persistent)
        self.map_points_pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.map_points_pcd)
        
        # Store all map points (persistent across frames)
        self.all_map_points = {}  # Dictionary: {landmark_id: position}
        
        # Line set for trajectory (blue path)
        self.trajectory_lines = o3d.geometry.LineSet()
        self.vis.add_geometry(self.trajectory_lines)
        
        # Current camera pose indicator (red sphere)
        self.current_camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        self.current_camera.paint_uniform_color([1, 0, 0])  # Red
        self.vis.add_geometry(self.current_camera)
        
        # Camera frustums (showing poses along trajectory)
        self.camera_frustums = []
        self.max_frustums = 30  # Keep last 30 camera poses
        
        # Grid floor
        self.grid_lines = None
        self.create_grid_floor()
        
        # View control settings
        self.follow_camera = False
        self.camera_position = np.array([0, 0, 0])
        
        # Configure rendering options
        render_option = self.vis.get_render_option()
        render_option.point_size = 3.0
        render_option.line_width = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        
        # Set initial camera view
        self.view_control = self.vis.get_view_control()
        self.view_control.set_zoom(0.3)
        # FIXED: Proper camera orientation
        self.view_control.set_front([0, 0, -1])  # Looking down negative Z
        self.view_control.set_lookat([0, 0, 0])
        self.view_control.set_up([0, -1, 0])  # Y is up in camera frame
        
        print("3D Viewer Controls:")
        print("  Mouse Left + Drag: Rotate view")
        print("  Mouse Right + Drag: Pan view")
        print("  Mouse Wheel: Zoom")
        print("  Press 'F' in viewer: Toggle Follow Camera")
        print("  Press 'R' in viewer: Reset view")
        
    def create_grid_floor(self, size=20, spacing=1.0):
        """Create a grid floor for reference - FIXED coordinate system"""
        if not HAS_OPEN3D:
            return
            
        lines = []
        points = []
        
        # Create grid lines on XZ plane (Y=0)
        for i in range(-size, size + 1):
            # Lines parallel to X axis
            points.append([i * spacing, 0, -size * spacing])
            points.append([i * spacing, 0, size * spacing])
            lines.append([len(points) - 2, len(points) - 1])
            
            # Lines parallel to Z axis
            points.append([-size * spacing, 0, i * spacing])
            points.append([size * spacing, 0, i * spacing])
            lines.append([len(points) - 2, len(points) - 1])
        
        self.grid_lines = o3d.geometry.LineSet()
        self.grid_lines.points = o3d.utility.Vector3dVector(points)
        self.grid_lines.lines = o3d.utility.Vector2iVector(lines)
        colors = np.tile([0.3, 0.3, 0.3], (len(lines), 1))  # Gray grid
        self.grid_lines.colors = o3d.utility.Vector3dVector(colors)
        
        self.vis.add_geometry(self.grid_lines)
        
    def add_map_points(self, points, landmark_ids=None):
        """Add new map points (persistent - don't disappear)"""
        if not HAS_OPEN3D or len(points) == 0:
            return
        
        # Add points to persistent storage
        if landmark_ids is not None and len(landmark_ids) == len(points):
            for i, lm_id in enumerate(landmark_ids):
                self.all_map_points[lm_id] = points[i]
        else:
            # If no IDs provided, use index
            start_id = len(self.all_map_points)
            for i, pt in enumerate(points):
                self.all_map_points[start_id + i] = pt
        
        # Update visualization
        self.update_map_points_visualization()
    
    def update_map_points_visualization(self):
        """Update the visualization of all map points - FIXED color mapping"""
        if not HAS_OPEN3D or len(self.all_map_points) == 0:
            return
        
        # Get all points
        points = np.array(list(self.all_map_points.values()))
        
        # Update point cloud
        self.map_points_pcd.points = o3d.utility.Vector3dVector(points)
        
        # FIXED: Color points based on height (Y coordinate - vertical in camera frame)
        colors = np.zeros((len(points), 3))
        y_coords = points[:, 1]  # Y is vertical
        y_min, y_max = y_coords.min(), y_coords.max()
        
        if y_max - y_min > 0.01:
            # Gradient from blue (low/floor) to red (high/ceiling)
            normalized_y = (y_coords - y_min) / (y_max - y_min)
            colors[:, 0] = normalized_y  # Red channel (high)
            colors[:, 2] = 1 - normalized_y  # Blue channel (low)
            colors[:, 1] = 0.3  # Small green component
        else:
            # All points same height - use cyan
            colors[:, 1] = 1.0  # Green
            colors[:, 2] = 1.0  # Blue
        
        self.map_points_pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.map_points_pcd)
        
    def clear_map_points(self):
        """Clear all map points"""
        self.all_map_points.clear()
        self.update_map_points_visualization()
        
    def update_trajectory(self, trajectory):
        """Update camera trajectory (blue line)"""
        if not HAS_OPEN3D or len(trajectory) < 2:
            return
            
        # Create line segments
        points = trajectory
        lines = [[i, i+1] for i in range(len(points)-1)]
        
        self.trajectory_lines.points = o3d.utility.Vector3dVector(points)
        self.trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
        
        # Blue trajectory
        colors = np.tile([0, 0.5, 1], (len(lines), 1))
        self.trajectory_lines.colors = o3d.utility.Vector3dVector(colors)
        
        self.vis.update_geometry(self.trajectory_lines)
        
    def update_current_camera(self, position):
        """Update current camera position (red sphere)"""
        if not HAS_OPEN3D:
            return
        
        self.camera_position = position
        
        # Move sphere to current position
        self.current_camera.translate(position - self.current_camera.get_center(), relative=False)
        self.vis.update_geometry(self.current_camera)
        
        # Follow camera if enabled
        if self.follow_camera:
            self.view_control.set_lookat(position)
            # Set camera behind and above current position (in camera frame coords)
            # Y is vertical, Z is forward
            cam_offset = np.array([0, 1, -2])  # Above and behind
            self.view_control.set_front([0, -0.3, 1])
    
    def add_camera_frustum(self, position, rotation):
        """Add a camera frustum at given pose"""
        if not HAS_OPEN3D:
            return
        
        # Limit number of frustums
        if len(self.camera_frustums) >= self.max_frustums:
            old_frustum = self.camera_frustums.pop(0)
            self.vis.remove_geometry(old_frustum, reset_bounding_box=False)
        
        # Create frustum
        frustum = self.create_camera_frustum(position, rotation)
        self.camera_frustums.append(frustum)
        self.vis.add_geometry(frustum, reset_bounding_box=False)
    
    def create_camera_frustum(self, position, rotation, size=0.1):
        """Create a camera frustum mesh"""
        # Define frustum in local camera coordinates
        # Camera looks along +Z with Y down
        half_width = size * 0.5
        half_height = size * 0.375  # 4:3 aspect
        depth = size * 0.5
        
        # Frustum corners in camera frame
        points = np.array([
            [0, 0, 0],  # Camera center
            [-half_width, -half_height, depth],  # Bottom-left
            [half_width, -half_height, depth],   # Bottom-right
            [half_width, half_height, depth],    # Top-right
            [-half_width, half_height, depth],   # Top-left
        ])
        
        # Transform to world frame
        points = (rotation @ points.T).T + position
        
        # Create line set
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
            [1, 2], [2, 3], [3, 4], [4, 1],  # Frame edges
        ]
        
        frustum = o3d.geometry.LineSet()
        frustum.points = o3d.utility.Vector3dVector(points)
        frustum.lines = o3d.utility.Vector2iVector(lines)
        colors = np.tile([0, 1, 0], (len(lines), 1))  # Green frustum
        frustum.colors = o3d.utility.Vector3dVector(colors)
        
        return frustum
    
    def toggle_follow_camera(self):
        """Toggle follow camera mode"""
        self.follow_camera = not self.follow_camera
        status = "ON" if self.follow_camera else "OFF"
        print(f"Follow Camera: {status}")
    
    def reset_view(self):
        """Reset camera view to default"""
        if not HAS_OPEN3D:
            return
        
        self.follow_camera = False
        self.view_control.set_zoom(0.3)
        self.view_control.set_front([0, 0, -1])
        self.view_control.set_lookat([0, 0, 0])
        self.view_control.set_up([0, -1, 0])
        print("View reset")
    
    def update(self):
        """Update visualization - FIXED: Non-blocking poll"""
        if not HAS_OPEN3D:
            return False
        
        # Use poll_events() and update_renderer() for non-blocking operation
        # This allows mouse interaction to work properly
        return self.vis.poll_events() and self.vis.update_renderer()
    
    def close(self):
        """Close visualization window"""
        if HAS_OPEN3D:
            self.vis.destroy_window()
    
    def save_map(self, filename="slam_map.ply"):
        """Save point cloud to PLY file"""
        if not HAS_OPEN3D or len(self.all_map_points) == 0:
            return False
        
        try:
            o3d.io.write_point_cloud(filename, self.map_points_pcd)
            print(f"Map saved: {filename}")
            return True
        except Exception as e:
            print(f"Error saving map: {e}")
            return False


class WebcamSLAM:
    """Main SLAM system with webcam input"""
    
    def __init__(self, camera_id=0, width=640, height=480, fps=30, use_3d_viewer=True):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        
        # Camera and SLAM components
        self.cap = None
        self.tracker = None
        
        # State
        self.frame_count = 0
        self.trajectory = []
        self.features_history = deque(maxlen=100)
        self.recent_observations = deque(maxlen=20)  # For triangulation
        
        # 3D Viewer
        self.use_3d_viewer = use_3d_viewer and HAS_OPEN3D
        self.viewer_3d = SLAM3DViewer() if self.use_3d_viewer else None
        
        # Camera parameters (estimated for webcam)
        fx = width * 0.8  # Focal length approximation
        fy = height * 0.8
        cx = width / 2
        cy = height / 2
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Status display
        self.status_text = "Initializing..."
        
    def initialize_camera(self):
        """Initialize camera"""
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized: {actual_width}x{actual_height}")
        
    def initialize_slam(self):
        """Initialize cuVSLAM tracker"""
        print("Initializing cuVSLAM tracker...")
        
        # Warm up GPU
        try:
            print("Warming up GPU...")
            cuvslam.warm_up_gpu()
            print("✓ GPU warmed up")
        except Exception as e:
            print(f"Warning: GPU warmup failed: {e}")
        
        try:
            # Get camera intrinsics
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # Create camera
            cam = cuvslam.Camera()
            cam.focal = [fx, fy]
            cam.principal = [cx, cy]
            cam.size = [self.width, self.height]
            
            # No distortion for webcam
            cam.distortion = cuvslam.Distortion(
                cuvslam.Distortion.Model.Brown,
                [0.0, 0.0, 0.0, 0.0, 0.0]
            )
            
            # Camera at rig origin
            cam.rig_from_camera = cuvslam.Pose(
                rotation=[0, 0, 0, 1],
                translation=[0, 0, 0]
            )
            
            # Create rig
            rig = cuvslam.Rig()
            rig.cameras = [cam]
            
            # Configure tracker
            cfg = cuvslam.Tracker.OdometryConfig()
            cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Mono
            cfg.enable_observations_export = True
            cfg.enable_final_landmarks_export = True  # For 3D map
            cfg.async_sba = False
            
            # Initialize tracker
            self.tracker = cuvslam.Tracker(rig, cfg)
            
            self.status_text = "SLAM Initialized"
            print(f"✓ cuVSLAM initialized with mode: {cfg.odometry_mode}")
            
        except Exception as e:
            print(f"Error initializing cuVSLAM: {e}")
            raise
        
    def track_frame(self, gray_image, timestamp_ns):
        """Track a single frame"""
        # Track with cuVSLAM - correct API: timestamp first, then frames list
        pose_estimate, _ = self.tracker.track(timestamp_ns, [gray_image])
        
        # Get observations for visualization
        observations = None
        try:
            observations = self.tracker.get_last_observations(0)
        except:
            pass
        
        # Update status based on tracking result
        if pose_estimate.world_from_rig is not None and pose_estimate.world_from_rig.pose is not None:
            self.status_text = "TRACKING"
        else:
            self.status_text = "INITIALIZING" if self.frame_count < 10 else "LOST"
        
        return pose_estimate, observations
    
    def draw_camera_feed(self, gray_image, pose_estimate, observations, fps):
        """Draw camera feed with overlays"""
        # Convert to color for display
        display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        h, w = display.shape[:2]
        
        # Check if we have a valid pose
        has_pose = (pose_estimate.world_from_rig is not None and 
                   pose_estimate.world_from_rig.pose is not None)
        
        # Draw status
        status_color = (0, 255, 0) if has_pose else (0, 0, 255)
        cv2.putText(display, f"Status: {self.status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Position info
        if has_pose:
            pose = pose_estimate.world_from_rig.pose
            pos = pose.translation
            cv2.putText(display, f"Pos: X={pos[0]:.2f} Y={pos[1]:.2f} Z={pos[2]:.2f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add to trajectory for map display
            if len(self.trajectory) == 0 or np.linalg.norm(np.array(pos) - np.array(self.trajectory[-1])) > 0.01:
                self.trajectory.append(pos)
        
        # Draw feature observations
        if observations is not None and len(observations) > 0:
            self.features_history.append(len(observations))
            
            for obs in observations:
                u, v = int(obs.u), int(obs.v)
                
                # Color based on tracking: Green = tracked landmark, Yellow = new
                if obs.id > 0:
                    color = (0, 255, 0)  # Green = tracked
                else:
                    color = (0, 255, 255)  # Yellow = new
                
                cv2.circle(display, (u, v), 3, color, -1)
                cv2.circle(display, (u, v), 5, color, 1)
            
            # Feature count
            cv2.putText(display, f"Features: {len(observations)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        cv2.putText(display, f"FPS: {fps:.1f}", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def draw_trajectory_map(self, size=800):
        """Draw top-down trajectory map"""
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        
        if len(self.trajectory) < 2:
            cv2.putText(canvas, "Building map...", (size//2 - 100, size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return canvas
        
        # Get trajectory points
        traj_array = np.array(self.trajectory)
        
        # Scale factor for display
        x_coords = traj_array[:, 0]
        z_coords = traj_array[:, 2]  # Use Z for depth
        
        if len(x_coords) > 0:
            x_range = x_coords.max() - x_coords.min()
            z_range = z_coords.max() - z_coords.min()
            scale = min(size * 0.8 / (x_range + 1e-6), size * 0.8 / (z_range + 1e-6))
            
            # Center offset
            x_offset = size // 2
            z_offset = size // 2
            
            # Draw trajectory
            for i in range(1, len(self.trajectory)):
                pt1 = self.trajectory[i-1]
                pt2 = self.trajectory[i]
                
                x1 = int(pt1[0] * scale + x_offset)
                z1 = int(-pt1[2] * scale + z_offset)  # Negative Z for proper orientation
                x2 = int(pt2[0] * scale + x_offset)
                z2 = int(-pt2[2] * scale + z_offset)
                
                # Color gradient based on height (Y)
                height = (pt2[1] - traj_array[:, 1].min()) / (traj_array[:, 1].max() - traj_array[:, 1].min() + 1e-6)
                color = (int(255 * (1 - height)), int(128), int(255 * height))
                
                cv2.line(canvas, (x1, z1), (x2, z2), color, 2)
            
            # Draw current position (larger red circle)
            current = self.trajectory[-1]
            x_cur = int(current[0] * scale + x_offset)
            z_cur = int(-current[2] * scale + z_offset)
            cv2.circle(canvas, (x_cur, z_cur), 8, (0, 0, 255), -1)
            cv2.circle(canvas, (x_cur, z_cur), 10, (255, 255, 255), 2)
        
        # Info text
        cv2.putText(canvas, "Top View (X-Z)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"Points: {len(self.trajectory)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(self.trajectory) > 0:
            distance = np.sum(np.linalg.norm(np.diff(traj_array, axis=0), axis=1))
            cv2.putText(canvas, f"Distance: {distance:.2f}m", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return canvas
    
    def update_3d_view(self, pose_estimate, observations):
        """Update 3D visualization"""
        if not self.use_3d_viewer:
            return
        
        try:
            # Check if we have a valid pose
            has_pose = (pose_estimate.world_from_rig is not None and 
                       pose_estimate.world_from_rig.pose is not None)
            
            # Update trajectory and camera
            if has_pose:
                pose = pose_estimate.world_from_rig.pose
                position = np.array(pose.translation)
                
                # Update trajectory line
                if len(self.trajectory) > 1:
                    self.viewer_3d.update_trajectory(self.trajectory)
                
                # Update current camera position
                self.viewer_3d.update_current_camera(position)
                
                # Add camera frustum occasionally
                if self.frame_count % 10 == 0:
                    rotation = rotation_matrix_from_quaternion(pose.rotation)
                    self.viewer_3d.add_camera_frustum(position, rotation)
            
            # Update map points from landmarks - FIXED: Use get_landmarks() plural
            # Get landmarks every 5 frames to build persistent map
            if self.frame_count % 5 == 0:
                try:
                    landmarks = self.tracker.get_landmarks()
                    if landmarks and len(landmarks) > 0:
                        points = []
                        lm_ids = []
                        for lm in landmarks:
                            points.append([lm.position[0], lm.position[1], lm.position[2]])
                            lm_ids.append(lm.id if hasattr(lm, 'id') else len(lm_ids))
                        
                        # Add to persistent map
                        self.viewer_3d.add_map_points(np.array(points), lm_ids)
                        
                        # Print occasionally
                        if self.frame_count % 30 == 0:
                            print(f"Map: {len(self.viewer_3d.all_map_points)} points")
                            
                except Exception as e:
                    # Fallback: triangulate from observations if landmarks unavailable
                    if observations and len(observations) > 0 and has_pose:
                        # Store observations with pose for triangulation
                        if not hasattr(self, 'recent_observations'):
                            self.recent_observations = deque(maxlen=20)
                        
                        self.recent_observations.append({
                            'obs': observations,
                            'pose': pose_estimate.world_from_rig.pose
                        })
                        
                        # Generate 3D points from recent observations every 10 frames
                        if self.frame_count % 10 == 0 and len(self.trajectory) > 5:
                            points = []
                            for obs_data in self.recent_observations:
                                cam_pos = np.array(obs_data['pose'].translation)
                                q = obs_data['pose'].rotation
                                R = rotation_matrix_from_quaternion(q)
                                
                                for obs in obs_data['obs']:
                                    if obs.id > 0:  # Only tracked features
                                        # Normalized image coordinates
                                        u_norm = (obs.u - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
                                        v_norm = (obs.v - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
                                        
                                        # Assume depth (rough approximation)
                                        depth = 1.5
                                        
                                        # Point in camera frame
                                        pt_cam = np.array([u_norm * depth, v_norm * depth, depth])
                                        
                                        # Transform to world
                                        pt_world = cam_pos + R @ pt_cam
                                        points.append(pt_world)
                            
                            if len(points) > 0:
                                self.viewer_3d.add_map_points(np.array(points))
            
            # CRITICAL: Use non-blocking update
            self.viewer_3d.update()
            
        except Exception as e:
            # Print errors occasionally for debugging
            if self.frame_count % 100 == 0:
                print(f"3D viewer update error: {e}")
    
    def save_map_data(self, filename="slam_data.pkl"):
        """Save complete SLAM data"""
        data = {
            'trajectory': self.trajectory,
            'map_points': dict(self.viewer_3d.all_map_points) if self.viewer_3d else {},
            'frame_count': self.frame_count
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"SLAM data saved: {filename}")
            
            # Also save point cloud in PLY format
            if self.viewer_3d:
                self.viewer_3d.save_map(filename.replace('.pkl', '.ply'))
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def load_map_data(self, filename="slam_data.pkl"):
        """Load complete SLAM data"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                
                self.trajectory = data.get('trajectory', [])
                
                if self.viewer_3d:
                    # Load map points
                    map_points = data.get('map_points', {})
                    self.viewer_3d.all_map_points = map_points
                    self.viewer_3d.update_map_points_visualization()
                
                print(f"SLAM data loaded: {filename}")
                print(f"  Trajectory: {len(self.trajectory)} points")
                print(f"  Map: {len(map_points)} points")
                return True
            else:
                print(f"File not found: {filename}")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def run(self):
        """Main SLAM loop"""
        try:
            # Initialize systems
            self.initialize_camera()
            self.initialize_slam()
            
            print("\n" + "="*60)
            print("SLAM Running! Controls:")
            print("  'q' or ESC - Quit")
            print("  's' - Save frame & map")
            print("  'r' - Reset SLAM")
            print("  'f' - Toggle Follow Camera (3D viewer)")
            print("  'v' - Reset 3D View")
            print("  'l' - Load saved map")
            print("  'c' - Clear map points")
            print("\nMove camera SLOWLY for best results!")
            print("="*60 + "\n")
            
            # Warm up - skip first few frames
            for _ in range(5):
                self.cap.read()
                time.sleep(0.05)
            
            # Timing for FPS
            fps_time = time.time()
            fps_counter = 0
            current_fps = 0
            
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame!")
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Generate timestamp (nanoseconds)
                timestamp_ns = int(time.time() * 1e9)
                
                # Track frame
                pose_estimate, observations = self.track_frame(gray, timestamp_ns)
                self.frame_count += 1
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_time > 1.0:
                    current_fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                
                # Draw all views
                camera_feed = self.draw_camera_feed(gray, pose_estimate, observations, current_fps)
                trajectory_map = self.draw_trajectory_map()
                
                # Show OpenCV windows
                cv2.imshow('CuVSLAM - Camera Feed', camera_feed)
                cv2.imshow('CuVSLAM - Trajectory Map', trajectory_map)
                
                # Update 3D viewer
                self.update_3d_view(pose_estimate, observations)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("\nStopping SLAM...")
                    break
                elif key == ord('s'):
                    # Save everything
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"slam_frame_{timestamp}.png", camera_feed)
                    self.save_map_data(f"slam_data_{timestamp}.pkl")
                    print(f"Saved frame and map data")
                elif key == ord('r'):
                    print("Resetting SLAM...")
                    self.trajectory.clear()
                    self.frame_count = 0
                    if self.viewer_3d:
                        self.viewer_3d.clear_map_points()
                    self.initialize_slam()
                elif key == ord('f'):
                    if self.viewer_3d:
                        self.viewer_3d.toggle_follow_camera()
                elif key == ord('v'):
                    if self.viewer_3d:
                        self.viewer_3d.reset_view()
                elif key == ord('l'):
                    self.load_map_data()
                elif key == ord('c'):
                    if self.viewer_3d:
                        self.viewer_3d.clear_map_points()
                        print("Map points cleared")
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        print("\nCleaning up...")
        
        # Auto-save on exit
        if len(self.trajectory) > 100:
            print("Auto-saving map before exit...")
            self.save_map_data("slam_data_autosave.pkl")
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.viewer_3d:
            self.viewer_3d.close()
        
        print(f"Total frames: {self.frame_count}")
        print(f"Trajectory points: {len(self.trajectory)}")
        if self.viewer_3d:
            print(f"Map points: {len(self.viewer_3d.all_map_points)}")
        print("Done!")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description='SLAM with ORB-SLAM3 Style Features')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--no-3d', action='store_true',
                       help='Disable 3D viewer')
    parser.add_argument('--load-map', type=str, default=None,
                       help='Load map from file on startup')
    
    args = parser.parse_args()
    
    # Create and run SLAM system
    slam_system = WebcamSLAM(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_3d_viewer=not args.no_3d
    )
    
    # Load map if specified
    if args.load_map:
        slam_system.load_map_data(args.load_map)
    
    slam_system.run()


if __name__ == "__main__":
    main()
