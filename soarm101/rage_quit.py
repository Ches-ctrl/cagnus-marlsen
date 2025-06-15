#!/usr/bin/env python

"""
Simple Robotic Arm Sweep Script

Executes a fast sequence with pauses:
1. Home position
2. Position 1 (FAST) → pause 2 seconds
3. Position 2 (FAST) → pause 2 seconds  
4. Home position

Usage: python sweep.py square_mapping_improved.json
"""

import json
import logging
import time
import sys
from typing import Dict

from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.common.utils.utils import init_logging


class SimpleSweeper:
    """Simple arm sweeper - just executes the sequence."""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.robot = None
        self.home_position = {}
        
        # Movement settings
        self.movement_duration = 0.8  # Fast movements
        self.pause_duration = 2.0     # 2 second pauses
        
        # Hardcoded positions from images
        self.position_1 = {
            'shoulder_pan.pos': -30.15,
            'shoulder_lift.pos': 96.10,
            'elbow_flex.pos': -79.98,
            'wrist_flex.pos': -9.00,
            'wrist_roll.pos': -48.07,
            'gripper.pos': 1.05
        }
        
        self.position_2 = {
            'shoulder_pan.pos': 16.30,
            'shoulder_lift.pos': 97.43,
            'elbow_flex.pos': -80.16,
            'wrist_flex.pos': -9.16,
            'wrist_roll.pos': -48.07,
            'gripper.pos': 1.05
        }
        
        self.load_home_position()
        self.setup_robot()
    
    def load_home_position(self):
        """Load home position from JSON file."""
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        self.home_position = data['metadata']['home_position']
        print(f"✅ Loaded home position from {self.json_file}")
    
    def setup_robot(self):
        """Initialize robot connection."""
        config = SO101FollowerConfig(
            port="/dev/ttyACM1",
            id="follower_arm", 
            use_degrees=False
        )
        self.robot = SO101Follower(config)
    
    def connect_robot(self):
        """Connect to robot."""
        print("🔌 Connecting to robot...")
        self.robot.connect(calibrate=False)
        print("✅ Connected!")
    
    def disconnect_robot(self):
        """Disconnect from robot."""
        if self.robot and self.robot.is_connected:
            self.robot.disconnect()
            print("🔌 Disconnected")
    
    def move_to_position(self, target_pos: Dict[str, float], description: str):
        """Move smoothly to target position."""
        print(f"🎯 {description}...")
        
        # Get current position
        current_pos = self.robot.get_observation()
        current_pos = {k: v for k, v in current_pos.items() if k.endswith('.pos')}
        
        # Smooth interpolation
        steps = int(self.movement_duration * 20)  # 20 steps per second
        step_delay = self.movement_duration / steps
        
        for i in range(steps + 1):
            t = i / steps
            t_smooth = t * t * (3.0 - 2.0 * t)  # Smoothstep
            
            waypoint = {}
            for joint in target_pos.keys():
                if joint in current_pos:
                    start_val = current_pos[joint]
                    end_val = target_pos[joint]
                    waypoint[joint] = start_val + t_smooth * (end_val - start_val)
                else:
                    waypoint[joint] = target_pos[joint]
            
            self.robot.send_action(waypoint)
            time.sleep(step_delay)
        
        print("✅ Reached!")
        if description != "Returning to HOME":
            print(f"⏸️  Pausing for {self.pause_duration} seconds...")
        time.sleep(self.pause_duration)
    
    def run_sweep(self):
        """Execute the complete sweep sequence."""
        print("\n🌊 Starting Fast Sweep Sequence")
        print("=" * 35)
        
        # Step 1: Home
        self.move_to_position(self.home_position, "Moving to HOME")
        
        # Step 2: Position 1 (fast + pause)
        self.move_to_position(self.position_1, "Moving FAST to POSITION 1")
        
        # Step 3: Position 2 (fast + pause)
        self.move_to_position(self.position_2, "Moving FAST to POSITION 2")
        
        # Step 4: Home (no pause)
        self.move_to_position(self.home_position, "Returning to HOME")
        
        print("\n✅ Fast sweep completed!")


def main():
    # Initialize logging
    init_logging()
    
    # Get JSON file
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "square_mapping.json"
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return
    
    # Create sweeper and run
    sweeper = SimpleSweeper(json_file)
    
    try:
        sweeper.connect_robot()
        sweeper.run_sweep()
    finally:
        sweeper.disconnect_robot()


if __name__ == "__main__":
    import os
    main()
