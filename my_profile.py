# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:06:56 2023

@author: ahstm
"""

import pandas as pd
import numpy as np
import collections


# 연결 리스트 노드 함수

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def printNodes(node:ListNode):
    crnt_node = node
    while crnt_node is not None:
        print(crnt_node.val , end= ' ')
        crnt_node = crnt_node.next