#!/usr/bin/env python
# -*- coding: utf-8 -*-

from promp.ros import ReplayableInteractiveProMP
import rospy

side = 'right'
dataset = 0

rospy.init_node('replay_interactive_promps')
promp = ReplayableInteractiveProMP(side, with_orientation=True, dataset_id=dataset)
promp.play()
promp.plot_demos()
