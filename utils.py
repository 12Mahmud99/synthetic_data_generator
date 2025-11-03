def graph_potential(positions,barrier_height,left_well,right_well,tilt):
    midpoint = (left_well + right_well)/2.0 
    a = abs(left_well-right_well)/2.0  
    y=barrier_height*(((positions - midpoint)**4)/(a**4) - 2*(positions - midpoint)**2/(a**2)) +tilt*(positions-midpoint)+barrier_height
    return y