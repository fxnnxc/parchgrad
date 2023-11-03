
def compute_in_out_ratio(R, xmin, xmax, ymin, ymax):
    assert R.min() >=0, f"{R.min()}"
    full_size = R.size(0) * R.size(1)
    size_in = (xmax- xmin) * (ymax- ymin)
    size_out = full_size - size_in
    
    if size_in ==0 or size_out ==0:
        return {"valid":False,        
                "size_in": size_in,
                "size_out" : size_out,
                "full_size" : full_size}
    r_in = R[ymin:ymax, xmin:xmax].sum()  
    r_tot = R.sum()
    r_out = r_tot - r_in
    
    mu_in = r_in / r_tot
    mu_in_w = mu_in * (full_size/size_in)
    
    mu_out = r_out / r_tot
    mu_out_w = mu_out * (full_size/size_out)

    output = {
        "valid":True,
        "mu_in": mu_in.item(),
        "mu_in_w": mu_in_w.item(),
        "mu_out": mu_out.item(),
        "mu_out_w": mu_out_w.item(),
        "r_in": r_in.item(),
        "r_out" : r_out.item(),
        "size_in": size_in,
        "size_out" : size_out,
        "full_size" : full_size
    }
    return output