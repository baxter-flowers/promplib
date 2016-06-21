#!/usr/bin/env python
# -*- coding: utf-8 -*-

from promp.interactive import InteractiveProMP
from baxter_commander import ArmCommander
from kinect2.client import Kinect2Client
import rospy

rospy.init_node('vocal_interactive_promps')

class VocalInteractiveProMPs(object):
    def __init__(self):
        # MOTION
        self.promp = InteractiveProMP('right')
        self.right = ArmCommander('right')
        self.init = self.right.get_current_state()

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
        self.kinect.speech.start()  # start with no callback to use the get() method
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
        self.kinect.tts.say('ready?')
        for countdown in [3, 2, 1, "go"]:
            rospy.sleep(1)
            self.kinect.tts.say('{}'.format(countdown))
        self.right.recorder.start()
        
        choice = ""
        while choice != 'stop':
            choice = self.read_user_input(['stop'])
             
        joints, eef = self.right.recorder.stop()
        self.kinect.tts.say('Motion recorded')
        try:
            self.promp.add_demonstration(joints)
        except ValueError:
            self.kinect.tts.say("Sorry I failed to record this demonstration")

    def set_goal(self):
        if self.promp.num_primitives > 0:
            self.kinect.tts.say('Move the robot and say ready to set the goal')

            choice = ""
            while choice != 'ready':
                choice = self.read_user_input(['ready'])

            goal_set = self.promp.set_goal(self.right.endpoint_pose())
            if goal_set:
                self.kinect.tts.say('I can reach this point, let me demonstrate')
                self.right.move_to_controlled(self.init)
                trajectory = self.promp.generate_trajectory()
                self.right.execute(trajectory, epsilon=99999)
            else:
                self.kinect.tts.say("I can't reach this point. {}".format(self.promp.status_writing))
        else:
            self.kinect.tts.say('The is no demonstration yet, please record at least one demo')

    def run(self):
        while not rospy.is_shutdown():
            self.right.move_to_controlled(self.init)
            if self.promp.num_primitives > 0:
                self.kinect.tts.say('Do you want to record a motion or set a new goal?')
                choice = self.read_user_input(['record', 'goal'])
                if choice == 'record':
                    self.record_motion()
                elif choice == 'goal':
                    self.set_goal()
            else:
                self.kinect.tts.say("Let's record a first demo, say stop to finish")
                rospy.sleep(4)
                self.record_motion()
            self.kinect.tts.say('There are {} primitive{} and {} demonstration{}'.format(self.promp.num_primitives,
                                                                                         's' if self.promp.num_primitives > 0 else '', self.promp.num_demos,
                                                                                         's' if self.promp.num_demos > 0 else ''))

VocalInteractiveProMPs().run()
