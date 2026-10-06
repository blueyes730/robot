# robot

Robot adapters publish standard ROS 2 sensor messages. Generic perception and
localization consume that interface independently, so changing robots, simulators,
hardware drivers, or replaying a rosbag does not require changing either node.

```text
robots/
├── nova_carter/
│   ├── config/sensors.{json,yaml}
│   ├── sensors.py
│   ├── spawn.py
│   └── isaacsim_ros_publishers.py
└── unitree_g1/
    ├── config/sensors.{json,yaml}
    ├── sensors.py
    └── spawn.py
ros_ws/src/
├── perception/
│   ├── package.xml, setup.py, setup.cfg, resource/perception
│   └── perception/{__init__.py,perception_node.py}
├── localization/
│   ├── package.xml, setup.py, setup.cfg, resource/localization
│   └── localization/{__init__.py,localization_node.py}
└── robot_bringup/
    ├── package.xml, setup.py, setup.cfg, resource/robot_bringup
    └── robot_bringup/__init__.py
simulation/isaacsim/scripts/
├── launch_nova_carter.py
└── launch_unitree_g1_sim.py
tests/
├── test_autonomy_nodes.py
└── test_isaacsim_ros_publishers.py
```

The Nova Carter adapter reuses the sensor handles and render products created by
`sensors.py`. Its launcher enables `isaacsim.ros2.bridge` and
`isaacsim.sensors.physics.nodes`, creates the sensors and ROS adapter, then starts
playback. Writer handles stay alive until simulation shutdown. Unitree ROS
publishing is a future adapter; its current implementation is unchanged.

| Published topic | Message type | Consumer |
| --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | perception |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | available for future consumers |
| `/lidar/points` | `sensor_msgs/msg/PointCloud2` | perception |
| `/imu/data` | `sensor_msgs/msg/Imu` | localization |

The generic nodes use sensor-data QoS and log metadata at most once every two
seconds per stream. They currently provide integration diagnostics; perception
algorithms and state estimation will be added later. `robot_bringup` contains
package metadata only.

Build on Ubuntu 24.04 with ROS 2 Jazzy:

```bash
cd ~/Projects/robot/ros_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Start Isaac Sim in a separate terminal. The installed `python.sh` configures the
bundled Jazzy bridge libraries by default:

```bash
cd ~/Projects/robot
~/isaacsim/python.sh simulation/isaacsim/scripts/launch_nova_carter.py
```

Add `--headless` to omit the window. The launcher defaults to Nova Carter's YAML
sensor config; select JSON with
`--sensor-config robots/nova_carter/config/sensors.json`. If using a system ROS
environment, source `/opt/ros/jazzy/setup.bash` before starting Isaac Sim and pass
`--no-ros-env` to `python.sh`. Keep `ROS_DOMAIN_ID` consistent across terminals.

Run each generic node in its own terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Projects/robot/ros_ws/install/setup.bash
ros2 run perception perception_node
```

```bash
source /opt/ros/jazzy/setup.bash
source ~/Projects/robot/ros_ws/install/setup.bash
ros2 run localization localization_node
```

Topic parameters (`camera_topic`, `lidar_topic`, `imu_topic`) and ordinary ROS
remapping allow the same nodes to consume another platform's drivers. For example:

```bash
ros2 run perception perception_node --ros-args \
  -r /camera/image_raw:=/robot/camera/image_raw \
  -r /lidar/points:=/robot/lidar/points \
  -p log_interval_sec:=5.0
ros2 run localization localization_node --ros-args -p imu_topic:=/robot/imu/data
```

Verify while simulation playback is active, using a sourced ROS terminal:

```bash
ros2 topic list
ros2 topic type /camera/image_raw
ros2 topic type /camera/camera_info
ros2 topic type /lidar/points
ros2 topic type /imu/data
ros2 topic hz /camera/image_raw
ros2 topic hz /lidar/points
ros2 topic hz /imu/data
```

Run the automated checks after building and sourcing the workspace:

```bash
cd ~/Projects/robot
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The adapter targets the locally installed Isaac Sim
`6.1.0-rc.26+release.49347.2d230af4.gl`. It uses the public sensor-runtime
`authoring_object.paths`, `render_product`, `attach_writer`, and `detach_writer`
APIs. The RGB writer name is derived from the `Rgb` render variable and ends in
`ROS2PublishImage`; it converts the renderer's RGBA output to RGB. Calibration
comes from `isaacsim.ros2.core.read_camera_info` and is passed to
`ROS2PublishCameraInfo`. RTX LiDAR uses `RtxLidarROS2PublishPointCloud`.

IMU publishing uses an on-demand OmniGraph driven by
`isaacsim.core.nodes.OnPhysicsStep`, reads the existing IMU prim through
`isaacsim.sensors.physics.IsaacReadIMU`, and connects valid readings to
`isaacsim.ros2.bridge.ROS2PublishImu`. It includes gravity, orientation, angular
velocity, and linear acceleration, and uses the sensor reading's timestamp.
The image, calibration, and point cloud writers also use simulation time.
The Isaac bridge handles ROS transport; the adapter does not initialize `rclpy`.

Default header frames are the existing prim names: `camera_sensor`,
`lidar_sensor`, and `imu_sensor`. `IsaacSimRosPublisherConfig` can override frames and
topics when integrating a TF tree. This milestone does not publish TF or `/clock`;
a future bringup layer should supply them when needed by estimation algorithms.
Camera calibration is read at adapter creation, so recreate the adapter after
changing camera intrinsics or resolution at runtime.

Physics remains at 200 Hz, rendering at 60 Hz, the camera at 30 Hz, and LiDAR at
10 Hz in simulation time. `ros2 topic hz` measures wall-time reception, so its
results depend on simulation speed and subscriber load.

The installed release candidate warns that the existing multi-tick LiDAR needs
motion BVH for motion effects. Enabling `enable_motion_bvh` at startup hangs in
this installation, and toggling the renderer setting after startup prevents
camera render products from running. The launcher retains its working renderer
configuration. ROS point clouds are delivered, but correct motion effects need
a compatible Isaac Sim renderer configuration before using scans on a moving
platform. The bundled robot asset also emits camera deprecation and articulation
metadata warnings; they did not prevent sensor publishing in the verified run.

Verification completed on the installed RTX 5080 with Jazzy: all three packages
built with `colcon build --symlink-install`, and all eight automated checks
passed. A headless Nova Carter run exposed all four topics with the expected
types. Both `ros2 run` executables logged live messages. A separate ROS subscriber
confirmed 640×480 RGB images, matching camera calibration, nonempty point clouds,
normalized IMU orientation, finite inertial measurements, and simulation
timestamps. The final CLI sample measured about 32.4 Hz for images, 9.8 Hz for
point clouds, and 229 Hz for IMU in wall time. Nominal delivery rates under load
remain unverified; best-effort sensor subscribers can drop messages.
