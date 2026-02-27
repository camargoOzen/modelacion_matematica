#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 06:41:22 2023

@author: carlosduquedaza
"""
import math
import numpy as np


secuencia3 = [1, 2, 4, 6, 9]
var1 = 0.0

for item in secuencia3:
    print(item)

print("\n")
for i in range(len(secuencia3)):
    var1 = float(secuencia3[i])**2 + math.pi
    print(f"La variable var1 es: {var1:.3f}")

print("\n")
for i in range(len(secuencia3)):
    var1 = np.sqrt(float(secuencia3[i])*0.3097)
    print(f"Ahora var1 = {var1:7.4e}")
    
