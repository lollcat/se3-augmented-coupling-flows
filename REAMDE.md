# Notes
- will need tricks for stability. 
- If we use zero init, then the flow is the identity transform which is fine.
- But for tests we don't use zero init, in which case we have to make the scale 
and shift very small to prevent massive changes to the initial points.