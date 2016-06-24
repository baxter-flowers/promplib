#!/usr/bin/env python
# -*- coding: utf-8 -*-

from promp.interactive import InteractiveProMP
from baxter_commander import ArmCommander
from kinect2.client import Kinect2Client
import rospy

rospy.init_node('vocal_interactive_promps')

class VocalInteractiveProMPs(object):
    def __init__(self, arm='left'):
        # MOTION
        self.promp = InteractiveProMP(arm)
        self.arm = ArmCommander(arm, ik='robot')
        self.init = self.arm.get_current_state()

        # INTERACTION (TTS + Speech recognition -- Kinect)
        rospy.loginfo("setting up the kinect...")
        self.kinect = Kinect2Client('BAXTERFLOWERS.local')
        self.kinect.tts.params.queue_on()
        self.kinect.tts.start()
        grammar = '''<grammar version="1.0" xml:lang="en-US" root="rootRule"
                              xmlns="http://www.w3.org/2001/06/grammar" tag-format="semantics/1.0">
                        <rule id="rootRule">
                            <one-of>
                                <item> Record a new motion
                                    <tag> out.action = "record"; </tag>
                                </item>
                                <item> Set a new goal
                                    <tag> out.action = "goal"; </tag>
                                </item>
                                <item> stop
                                    <tag> out.action = "stop"; </tag>
                                </item>
                                <item> ready
                                    <tag> out.action = "ready"; </tag>
                                </item>
                            </one-of>
                        </rule>
                     </grammar>'''
        self.kinect.speech.params.set_grammar(grammar)
        self.kinect.display_speech()
        success = self.kinect.speech.start()  # start with no callback to use the get() method
        assert success == '', success
        rospy.loginfo("Kinect started!")

    def read_user_input(self, all_semantics):
        speech = self.kinect.speech.get()
        if all_semantics is not None and speech is not None:
            if 'semantics' in speech:
                word = speech['semantics'][0]  # TODO we might have received several words?
                if word in all_semantics:
                    return word
        return ""

    def record_motion(self):
        for countdown in ['ready?', 3, 2, 1, "go"]:
            self.say('{}'.format(countdown), blocking=False)
            rospy.sleep(1)
        self.arm.recorder.start(10)
        rospy.loginfo("Recording...")

        choice = ""
        while choice != 'stop':
            choice = self.read_user_input(['stop'])

        joints, eef = self.arm.recorder.stop()
        self.say('Motion recorded, please wait...', blocking=False)
        try:
            self.promp.add_demonstration(joints)
        except ValueError:
            self.say("Sorry I failed to record this demonstration")

    def set_goal(self):
        if self.promp.num_primitives > 0:
            self.say('Move the robot and say ready to set the goal')

            choice = ""
            while choice != 'ready':
                choice = self.read_user_input(['ready'])

            goal_set = self.promp.set_goal(self.arm.endpoint_pose())
            if goal_set:
                self.say('I can reach this object, let me demonstrate', blocking=False)
                self.arm.translate_to_cartesian([0, 0, 0.2], 'base', 2)
                self.arm.move_to_controlled(self.init)
                self.arm.open()
                trajectory = self.promp.generate_trajectory()
                self.arm.execute(trajectory)
                self.arm.close()
                self.arm.translate_to_cartesian([0, 0, 0.2], 'base', 2)
                if self.arm.gripping():
                    rospy.loginfo('Take it!')
                    rospy.sleep(3)
                    self.arm.open()
            else:
                self.say("I don't know how to reach this object. {}".format(self.promp.status_writing))
        else:
            self.say('The is no demonstration yet, please record at least one demo')

    def say(self, what, blocking=False):
        rospy.loginfo(what)
        self.kinect.tts.say(what, True)
        rospy.logwarn('BLOCKING')

    def run(self):
        while not rospy.is_shutdown():
            self.arm.move_to_controlled(self.init)
            if self.promp.num_primitives > 0:
                self.say('Do you want to record a motion or set a new goal?')
                choice = self.read_user_input(['record', 'goal'])
                if choice == 'record':
                    self.record_motion()
                elif choice == 'goal':
                    self.set_goal()
            else:
                self.say("Let's record a first demo, say stop to finish")
                rospy.sleep(4)
                self.record_motion()
            self.say('There are {} primitive{} and {} demonstration{}'.format(self.promp.num_primitives,
                                                                                         's' if self.promp.num_primitives > 0 else '', self.promp.num_demos,
                                                                                         's' if self.promp.num_demos > 0 else ''))

VocalInteractiveProMPs().run()
