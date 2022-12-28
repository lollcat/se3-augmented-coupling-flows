# Notes
**need tricks for stability** 
- If we use zero init, then the flow is the identity transform which is fine.
- But for tests we don't use zero init, in which case we have to 
(1) use 64 bit, and 
(2) make the scale and shift very small to prevent massive changes to the initial points.