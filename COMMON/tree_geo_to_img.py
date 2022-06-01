import numpy as np
import math
import os
from PIL import Image, ImageFilter
import json

def flatten(t):
    return [item for sublist in t for item in sublist]

def mapFromTo(x,a,b,c,d):
   y=(x-a)/(b-a)*(d-c)+c
   return y

def smoothen_rads(R,n0,n1):
    R_new = np.copy(R)
    for i in range(R.shape[0]):
        for j in range(360):
            near_rads = []
            for a in range(-n0,n0+1):
                for b in range(-n1,n1+1):
                    ii = (i+a)%R.shape[0]
                    jj = (j+b)%360
                    near_rads.append(R[ii][jj])
            rad = sum(near_rads)/len(near_rads)
            R_new[i][j] = rad
    return R_new

def get_min_max_radii_and_tree_height(path):
    f = open(path)
    rads = json.load(f)
    rads_flat = flatten(rads)
    f.close()
    return min(rads_flat), max(rads_flat), len(rads)*10

def load_and_save_pith_and_outer_shape(path_pith, path_rad, filename, min_r, max_r, show=True):

    #pith
    f = open(path_pith)
    piths = json.load(f)
    znum = len(piths)
    pith_and_rads = np.zeros((znum,360,3), dtype='float64')
    mid_index = int(0.5*len(piths))
    x0 = piths[mid_index][0]
    y0 = piths[mid_index][1]

    for i in range(znum):
        x = piths[i][0]-x0
        y = piths[i][1]-y0
        x = mapFromTo(x,-0.5*min_r,0.5*min_r,0,255)
        y = mapFromTo(y,-0.5*min_r,0.5*min_r,0,255)
        for j in range(360):
            pith_and_rads[i][j][0]=x #red
            pith_and_rads[i][j][1]=y #green
    f.close()

    #outer shape rads
    f = open(path_rad)
    rads = json.load(f)
    rads = np.array(rads)
    for i in range(znum):
        for j in range(360):
            rads[i][j] = mapFromTo(rads[i][j]-min_r,0,max_r-min_r,0,255)
    rads = smoothen_rads(rads,5,2)
    for i in range(znum):
        for j in range(360):
            pith_and_rads[i][j][2]=rads[i][j] #blue
    f.close()

    #image
    img = Image.fromarray(np.uint8(pith_and_rads) , 'RGB')
    img = img.resize((180, znum))
    img.save(filename)
    img.show()

def load_and_save_knot_params_to_hmap_rmap(path,filename_hmap,filename_rmap,filename_smap,min_r,max_r,max_h,col_width=128,row_height=1):

    f = open(path)
    knot_params = json.load(f)
    f.close()
    k_num = len(knot_params)

    width = col_width; #in pixels
    knot_height = np.zeros((row_height*k_num,width,3))
    knot_rotation = np.zeros((row_height*k_num,width,3))
    knot_state = np.zeros((row_height*k_num,width,3))

    for i in range(k_num):
        A,B,C,D,E,F,G,H,I = knot_params[i]

        if I<0.1: continue # Dies super early

        H_mapped = mapFromTo(H, 0,max_r,0,width) # end distance
        I_mapped = mapFromTo(I, 0,max_r,0,width) # death distance

        for j in range(width):
            rp = mapFromTo(j,0,width,0,max_r) # distnace from pith

            #height
            z_start = C
            z_fine = D*math.sqrt(rp)+E*rp
            z_fine_up = 0.0
            z_fine_dn = 0.0
            if z_fine>0: z_fine_up = z_fine
            else: z_fine_dn = -z_fine

            #rotation
            om_start = (F+2*math.pi)%2*math.pi
            if j==0: om_twist=0.0
            else: om_twist = G*math.log(rp)
            om_twist_ccw = 0.0
            om_twist_cw = 0.0
            if om_twist>0: om_twist_ccw = om_twist
            else: om_twist_cw = -om_twist

            for k in range(row_height):

                ## height
                knot_height[i*row_height+k][j][0] = mapFromTo(z_start,0,max_h,0,255)          #red   = start
                knot_height[i*row_height+k][j][1] = mapFromTo(z_fine_up,0.0,min_r,0,255)      #green = up
                knot_height[i*row_height+k][j][2] = mapFromTo(z_fine_dn,0.0,min_r,0,255)      #blue  = down

                ## rotation
                knot_rotation[i*row_height+k][j][0] = mapFromTo(om_start, 0,2.0*math.pi, 0,255)      #red   = start
                knot_rotation[i*row_height+k][j][1] = mapFromTo(om_twist_ccw, 0,0.5*math.pi, 0,255)  #green = left twist
                knot_rotation[i*row_height+k][j][2] = mapFromTo(om_twist_cw, 0,0.5*math.pi, 0,255)   #blue  = right twist

                ## state
                if j<I_mapped: # alive
                    knot_state[i*row_height+k][j][0] = 255;    #red   = alive
                knot_state[i*row_height+k][j][1] = mapFromTo(I,0.0,max_r,0,255)    #green = dead
                #knot_state[i*row_height+k][j][2] = mapFromTo(H,0.0,max_r,0,255)    #blue  = broken off




    img = Image.fromarray(np.uint8(knot_height) , 'RGB')
    img.save(filename_hmap)
    #img.show()

    img = Image.fromarray(np.uint8(knot_rotation) , 'RGB')
    img.save(filename_rmap)
    #img.show()

    img = Image.fromarray(np.uint8(knot_state) , 'RGB')
    img.save(filename_smap)
    #img.show()

    return k_num

tree_name = "Gran_1"
tree_geo_maps_path = os.path.abspath('..') + "\\tree_geo_maps"

pith_path = "tree_data\\"+tree_name+"_pith.json"
radii_path = "tree_data\\"+tree_name+"_radii.json"
knot_path =  "tree_data\\"+tree_name+"_knots.json"

fname_pmap = tree_geo_maps_path+"\\pith_and_radius_map.bmp"
fname_hmap = tree_geo_maps_path+"\\knot_height_map.bmp"
fname_rmap = tree_geo_maps_path+"\\knot_orientation_map.bmp"
fname_smap = tree_geo_maps_path+"\\knot_state_map.bmp"
fname_parms = tree_geo_maps_path+"\\map_params.json"

min_rad, max_rad, max_height = get_min_max_radii_and_tree_height(radii_path)

print("Min radius:", min_rad, "mm")
print("Max radius:", max_rad, "mm")
print("Tree height", max_height, "mm")

load_and_save_pith_and_outer_shape(pith_path, radii_path, fname_pmap, min_rad, max_rad, show=False)

num_knots = load_and_save_knot_params_to_hmap_rmap(knot_path, fname_hmap, fname_rmap, fname_smap, min_rad, max_rad, max_height, col_width=32, row_height=4)

print("Number of knots", num_knots)

with open(fname_parms, 'w') as f:
    json.dump([min_rad, max_rad, max_height, num_knots], f)
