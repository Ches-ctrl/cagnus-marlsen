#!/usr/bin/env python

"""
Chess Move Executor for 6DoF Robotic Arm

This script executes complete chess moves by picking up pieces from source squares
and placing them on destination squares using a specific sequence with stabilization
pauses for improved reliability.

Key Feature: Overrides wrist_roll and gripper positions from the JSON calibration
data to maintain precise control during chess moves, while using the arm joint
positions (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex) from calibration.

Stabilization: Includes configurable pauses before grabbing and placing pieces
to ensure the arm is stable and not vibrating.

Usage:
python chess_move_executor.py square_mapping_improved.json

Move format: e7e5 (from e7 to e5)
"""

import json
import logging
import time
import sys
import re
from typing import Dict, Optional, Tuple
from datetime import datetime

from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.common.utils.utils import init_logging


class ChessMoveExecutor:
    """
    Execute complete chess moves with pick-and-place operations.
    """
    
    def __init__(self, json_file: str):
        """
        Initialize the chess move executor.
        
        Args:
            json_file: Path to the improved square positions JSON file
        """
        self.json_file = json_file
        self.positions = {}
        self.metadata = {}
        self.robot = None
        self.home_position = {}
        
        # Movement configuration
        self.gripper_open = 15.0      # Open gripper position
        self.gripper_closed = 1.5     # Closed gripper position
        self.pickup_wrist_roll = -55.0  # Wrist roll for pickup
        self.final_wrist_roll = 0.0    # Final wrist roll position
        self.movement_duration = 2.0   # Duration for smooth movements
        self.pause_duration = 0.5      # Pause between operations
        self.stabilization_pause = 1.0 # Pause for arm stabilization before grab/place
        
        # Note: We override wrist_roll and gripper positions from JSON file
        # to maintain precise control during chess moves
        
        self.load_positions()
        self.setup_robot()
    
    def load_positions(self):
        """Load positions from the improved JSON file."""
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            self.metadata = data.get('metadata', {})
            self.positions = data.get('positions', {})
            
            print(f"✅ Loaded {len(self.positions)} improved positions from {self.json_file}")
            
            # Show anchor points used
            anchor_squares = self.metadata.get('anchor_squares', [])
            if anchor_squares:
                print(f"📍 Using {len(anchor_squares)} calibrated anchor points: {', '.join([s.upper() for s in anchor_squares])}")
            
        except FileNotFoundError:
            print(f"❌ Position file not found: {self.json_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON file: {e}")
            raise
    
    def setup_robot(self):
        """Initialize robot connection."""
        config = SO101FollowerConfig(
            port="/dev/ttyACM1",
            id="follower_arm",
            use_degrees=False
        )
        self.robot = SO101Follower(config)
    
    def connect_robot(self):
        """Connect to the robot and load home position."""
        try:
            print("🔌 Connecting to SO101 follower arm...")
            self.robot.connect(calibrate=False)
            print("✅ Successfully connected to robot")
            
            # Load home position from metadata
            self.home_position = self.metadata.get('home_position', {})
            if self.home_position:
                print("🏠 Home position loaded from metadata:")
                for joint, value in self.home_position.items():
                    joint_name = joint.replace('.pos', '').replace('_', ' ').title()
                    print(f"  {joint_name:15}: {value:7.2f}°")
            else:
                print("❌ No home position in metadata!")
                print("💡 Please set home position using the original piece_mover.py script")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to robot: {e}")
            return False
    
    def disconnect_robot(self):
        """Disconnect from the robot."""
        try:
            if self.robot and self.robot.is_connected:
                self.robot.disconnect()
                print("🔌 Robot disconnected")
        except Exception as e:
            print(f"⚠️  Error disconnecting robot: {e}")
    
    def get_current_robot_position(self) -> Dict[str, float]:
        """Get current joint positions from robot."""
        try:
            observation = self.robot.get_observation()
            return {k: v for k, v in observation.items() if k.endswith('.pos')}
        except Exception as e:
            print(f"⚠️  Could not read robot position: {e}")
            return {}
    
    def move_to_position(self, target_pos: Dict[str, float], duration: float = None):
        """Execute smooth movement to target position."""
        if duration is None:
            duration = self.movement_duration
            
        current_pos = self.get_current_robot_position()
        if not current_pos:
            # Fallback to direct movement
            self.robot.send_action(target_pos)
            time.sleep(1.0)
            return
        
        steps = int(duration * 20)  # 20 steps per second
        step_delay = duration / steps
        
        for i in range(steps + 1):
            t = i / steps
            t_smooth = t * t * (3.0 - 2.0 * t)  # Smoothstep interpolation
            
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
    
    def move_to_home(self, preserve_wrist_gripper: bool = True):
        """
        Move robot to home position.
        
        Args:
            preserve_wrist_gripper: If True, keeps current wrist/gripper positions
        """
        print("🏠 Moving to home position...")
        
        target_pos = self.home_position.copy()
        
        if preserve_wrist_gripper:
            # Keep current wrist and gripper positions, don't use home values
            current_pos = self.get_current_robot_position()
            if current_pos:
                if 'wrist_roll.pos' in current_pos:
                    target_pos['wrist_roll.pos'] = current_pos['wrist_roll.pos']
                if 'gripper.pos' in current_pos:
                    target_pos['gripper.pos'] = current_pos['gripper.pos']
        
        self.move_to_position(target_pos)
        time.sleep(self.pause_duration)
    
    def move_to_square(self, square: str, description: str = "", 
                      wrist_roll: Optional[float] = None, 
                      gripper: Optional[float] = None):
        """
        Move robot to a specific chess square with optional wrist/gripper override.
        
        Args:
            square: Chess square to move to
            description: Optional description for logging
            wrist_roll: Override wrist_roll position (if None, uses current position)
            gripper: Override gripper position (if None, uses current position)
        """
        square = square.lower()
        
        if square not in self.positions:
            raise ValueError(f"Square '{square}' not found in positions")
        
        # Get base position from JSON (arm joints only)
        target_pos = self.positions[square].copy()
        
        # Override wrist_roll and gripper with custom values or current position
        current_pos = self.get_current_robot_position()
        
        if wrist_roll is not None:
            target_pos['wrist_roll.pos'] = wrist_roll
        elif current_pos and 'wrist_roll.pos' in current_pos:
            # Keep current wrist position instead of using JSON value
            target_pos['wrist_roll.pos'] = current_pos['wrist_roll.pos']
        
        if gripper is not None:
            target_pos['gripper.pos'] = gripper
        elif current_pos and 'gripper.pos' in current_pos:
            # Keep current gripper position instead of using JSON value
            target_pos['gripper.pos'] = current_pos['gripper.pos']
        
        if description:
            print(f"🎯 Moving to square {square.upper()} {description}...")
        else:
            print(f"🎯 Moving to square {square.upper()}...")
        
        self.move_to_position(target_pos)
        time.sleep(self.pause_duration)
    
    def set_gripper(self, position: float, description: str = ""):
        """Set gripper to specific position."""
        current_pos = self.get_current_robot_position()
        if not current_pos:
            print("❌ Could not read current position for gripper control")
            return
        
        target_pos = current_pos.copy()
        target_pos['gripper.pos'] = position
        
        if description:
            print(f"🤏 {description} (gripper: {position})...")
        else:
            print(f"🤏 Setting gripper to {position}...")
        
        self.move_to_position(target_pos, duration=1.0)
        time.sleep(self.pause_duration)
    
    def stabilize_arm(self, description: str = "Stabilizing arm"):
        """Pause to allow arm to stabilize and stop vibrating."""
        print(f"⏸️  {description} (pausing {self.stabilization_pause}s)...")
        time.sleep(self.stabilization_pause)
    
    def set_wrist_roll(self, angle: float, description: str = ""):
        """Set wrist roll to specific angle."""
        current_pos = self.get_current_robot_position()
        if not current_pos:
            print("❌ Could not read current position for wrist control")
            return
        
        target_pos = current_pos.copy()
        target_pos['wrist_roll.pos'] = angle
        
        if description:
            print(f"🔄 {description} (wrist roll: {angle}°)...")
        else:
            print(f"🔄 Setting wrist roll to {angle}°...")
        
        self.move_to_position(target_pos, duration=1.0)
        time.sleep(self.pause_duration)
    
    def parse_move(self, move_str: str) -> Tuple[str, str]:
        """
        Parse chess move string into source and destination squares.
        
        Args:
            move_str: Move in format like "e7e5" or "e7-e5"
            
        Returns:
            Tuple of (source_square, dest_square)
        """
        # Remove any spaces and convert to lowercase
        move_str = move_str.replace(" ", "").replace("-", "").lower()
        
        # Match pattern like "e7e5"
        match = re.match(r'^([a-h][1-8])([a-h][1-8])$', move_str)
        if not match:
            raise ValueError(f"Invalid move format: '{move_str}'. Expected format: 'e7e5'")
        
        source_square = match.group(1)
        dest_square = match.group(2)
        
        return source_square, dest_square
    
    def execute_move(self, move_str: str):
        """
        Execute a complete chess move with the specified sequence.
        
        Args:
            move_str: Move in format like "e7e5"
        """
        try:
            # Parse the move
            source_square, dest_square = self.parse_move(move_str)
            
            print(f"\n♟️  Executing chess move: {source_square.upper()} → {dest_square.upper()}")
            print("=" * 50)
            
            # Step 1: Move to home position (reset everything to start fresh)
            self.move_to_home(preserve_wrist_gripper=False)
            
            # Step 2: Set gripper to open position (15)
            self.set_gripper(self.gripper_open, "Opening gripper")
            
            # Step 3: Rotate wrist to pickup position (-55°)
            self.set_wrist_roll(self.pickup_wrist_roll, "Setting pickup wrist angle")
            
            # Step 4: Move to source square (with pickup wrist angle and open gripper)
            self.move_to_square(source_square, "(pickup position)", 
                              wrist_roll=self.pickup_wrist_roll, 
                              gripper=self.gripper_open)
            
            # Step 4.5: Stabilize before grabbing
            self.stabilize_arm("Stabilizing before grabbing piece")
            
            # Step 5: Close gripper to grab piece (1.5)
            self.set_gripper(self.gripper_closed, "Closing gripper to grab piece")
            
            # Step 6: Return to home (preserve wrist angle and closed gripper)
            self.move_to_home(preserve_wrist_gripper=True)
            
            # Step 7: Move to destination square (with pickup wrist angle and closed gripper)
            self.move_to_square(dest_square, "(drop position)", 
                              wrist_roll=self.pickup_wrist_roll, 
                              gripper=self.gripper_closed)
            
            # Step 7.5: Stabilize before placing
            self.stabilize_arm("Stabilizing before placing piece")
            
            # Step 8: Open gripper to release piece (12)
            self.set_gripper(self.gripper_open, "Opening gripper to release piece")
            
            # Step 9: Return to home (preserve wrist angle and open gripper)
            self.move_to_home(preserve_wrist_gripper=True)
            
            # Step 10: Set wrist roll to final position (0°)
            self.set_wrist_roll(self.final_wrist_roll, "Resetting wrist to neutral")
            
            print(f"✅ Move {source_square.upper()} → {dest_square.upper()} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error executing move: {e}")
            print("🏠 Attempting to return to safe home position...")
            try:
                self.move_to_home(preserve_wrist_gripper=False)
                self.set_wrist_roll(self.final_wrist_roll, "Resetting wrist to neutral")
            except:
                print("⚠️  Could not return to home position safely")
            raise
    
    def show_configuration(self):
        """Show current configuration settings."""
        print("\n⚙️  Current Configuration:")
        print(f"  Gripper Open:       {self.gripper_open}")
        print(f"  Gripper Closed:     {self.gripper_closed}")
        print(f"  Pickup Wrist:       {self.pickup_wrist_roll}°")
        print(f"  Final Wrist:        {self.final_wrist_roll}°")
        print(f"  Movement Duration:  {self.movement_duration}s")
        print(f"  Pause Duration:     {self.pause_duration}s")
        print(f"  Stabilization:      {self.stabilization_pause}s")
    
    def interactive_mode(self):
        """Interactive mode for executing chess moves."""
        print("\n♟️  Interactive Chess Move Executor")
        print("=" * 40)
        print("Commands:")
        print("  <move>       - Execute move (e.g., 'e7e5', 'Ng1f3')")
        print("  home         - Move to home position")
        print("  config       - Show configuration")
        print("  test         - Test gripper and wrist movements")
        print("  stabilize X  - Set stabilization pause to X seconds")
        print("  quit         - Exit")
        print()
        self.show_configuration()
        
        while True:
            try:
                command = input("\n[CHESS] Command: ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("👋 Returning to home and exiting...")
                    self.move_to_home(preserve_wrist_gripper=False)
                    self.set_wrist_roll(self.final_wrist_roll)
                    break
                
                elif command.lower() == 'home':
                    self.move_to_home(preserve_wrist_gripper=False)
                
                elif command.lower() == 'config':
                    self.show_configuration()
                
                elif command.lower() == 'test':
                    self.test_movements()
                
                elif command.lower().startswith('stabilize '):
                    try:
                        parts = command.split()
                        if len(parts) == 2:
                            new_pause = float(parts[1])
                            if 0.1 <= new_pause <= 5.0:
                                self.stabilization_pause = new_pause
                                print(f"✅ Stabilization pause set to {new_pause}s")
                            else:
                                print("❌ Stabilization pause must be between 0.1 and 5.0 seconds")
                        else:
                            print("❌ Usage: stabilize <seconds> (e.g., 'stabilize 1.5')")
                    except ValueError:
                        print("❌ Invalid number. Usage: stabilize <seconds> (e.g., 'stabilize 1.5')")
                
                elif len(command) >= 4:
                    # Try to parse as a chess move
                    try:
                        self.execute_move(command)
                    except ValueError as e:
                        print(f"❌ {e}")
                        print("💡 Use format like 'e7e5' for moves")
                
                else:
                    print(f"❌ Unknown command: '{command}'")
                    print("   Try: <move> (e.g. 'e7e5'), 'home', 'config', 'test', or 'quit'")
            
            except KeyboardInterrupt:
                print("\n⏹️  Interrupted")
                self.move_to_home(preserve_wrist_gripper=False)
                self.set_wrist_roll(self.final_wrist_roll)
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_movements(self):
        """Test gripper and wrist movements."""
        print("\n🧪 Testing gripper and wrist movements...")
        
        try:
            # Test gripper
            print("Testing gripper...")
            self.set_gripper(self.gripper_open, "Opening gripper")
            time.sleep(0.5)
            self.set_gripper(self.gripper_closed, "Closing gripper")
            time.sleep(0.5)
            self.set_gripper(self.gripper_open, "Opening gripper")
            
            # Test wrist
            print("Testing wrist roll...")
            self.set_wrist_roll(self.pickup_wrist_roll, "Setting pickup angle")
            time.sleep(0.5)
            self.set_wrist_roll(self.final_wrist_roll, "Resetting to neutral")
            
            print("✅ Test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    def run(self):
        """Main execution function."""
        print("🤖 Chess Move Executor")
        print("=" * 30)
        print(f"📁 Using: {self.json_file}")
        print(f"📍 Positions: {len(self.positions)} squares")
        
        # Show improvement info
        if 'improved_at' in self.metadata:
            print(f"✨ Improved positions from: {self.metadata['improved_at'][:10]}")
        
        print()
        
        # Connect to robot
        if not self.connect_robot():
            return
        
        try:
            # Move to home position initially
            print("🏠 Moving to initial home position...")
            self.move_to_home(preserve_wrist_gripper=False)
            self.set_wrist_roll(self.final_wrist_roll, "Setting initial wrist position")
            
            # Start interactive mode
            self.interactive_mode()
            
        finally:
            self.disconnect_robot()


def main():
    """Main function."""
    # Initialize logging
    init_logging()
    
    # Get JSON file from command line or use default
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Look for improved JSON files
        import os
        candidates = [
            "square_mapping.json",
        ]
        
        json_file = None
        for candidate in candidates:
            if os.path.exists(candidate):
                json_file = candidate
                break
        
        if not json_file:
            print("❌ No position file found.")
            print("Usage: python chess_move_executor.py <positions.json>")
            print("Or place 'square_mapping_improved.json' in current directory")
            return
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return
    
    # Create and run executor
    executor = ChessMoveExecutor(json_file)
    executor.run()


if __name__ == "__main__":
    main()
