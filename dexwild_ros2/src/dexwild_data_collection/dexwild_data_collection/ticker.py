# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

'''
This node is for regulating the rate of the data collection and synchronization process.
'''
class TickerNode(Node):
    def __init__(self):
        super().__init__('ticker_node')
        self.declare_parameter('rate', 30.0)  # default 30 Hz

        rate = self.get_parameter('rate').get_parameter_value().double_value
        self.pub = self.create_publisher(PoseStamped, '/ticker', 10)
        timer_period = 1.0 / rate

        self.timer = self.create_timer(timer_period, self.tick)
        self.get_logger().info(f'Ticker started at {rate} Hz')

    def tick(self):
        empty_message = PoseStamped()
        empty_message.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(empty_message)

def main(args=None):
    rclpy.init(args=args)
    node = TickerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()
