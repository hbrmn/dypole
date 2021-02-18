# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:30:31 2021

@author: HB
"""

# What guis are all about - data processing

# Architecture - signals and slots (compare with Qt documentation)

# Signals . e.g. clicked() pressed() released() toggled()
# Slots e.g. setChecked setIconSize() clear() copy() undo()
# All widgets have all slots and widgets



class MyGUI:
    
    def __init__(self):
        b = button()
        l = line_edits()
        
        when b.is_clicked()
        
    def do_processing(self):
        value = l.text()
        process(value)
        
while True: #event loop
    event = get_event()
    process(event, MyGui)
        
        
        
        
        