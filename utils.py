def graph_potential(positions,barrier_height,left_well,right_well,tilt):
    midpoint = (left_well + right_well)/2.0 
    a = abs(left_well-right_well)/2.0  
    y=barrier_height*(((positions - midpoint)**4)/(a**4) - 2*(positions - midpoint)**2/(a**2)) +tilt*(positions-midpoint)+barrier_height
    return y

def graph_potential_classic(positions,curvature, well_separation):
    return curvature*(positions**2 - well_separation**2)**2