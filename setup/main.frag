#version 330
#define PI 3.1415926538

in vec3 outTexCoords;
in vec3 normal;
in vec3 fragPos;
in float zrat;

uniform float rmin;
uniform float rmax;
uniform float hmax;
uniform int N;
uniform sampler2D ColorMap;
uniform sampler2D SpecularMap;
uniform sampler2D NormalMap;
uniform sampler2D PithRadiusMap;
uniform sampler2D KnotHeightMap;
uniform sampler2D KnotOrientMap;
uniform sampler2D KnotStateMap;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform mat4 model;
uniform mat4 view;
uniform float time;

out vec4 outColor;

vec4 cubic(float x){
    float x2 = x * x;
    float x3 = x2 * x;
    vec4 w;
    w.x =   -x3 + 3*x2 - 3*x + 1;
    w.y =  3*x3 - 6*x2       + 4;
    w.z = -3*x3 + 3*x2 + 3*x + 1;
    w.w =  x3;
    return w / 6.f;
  }

vec4 textureBicubic(sampler2D sampler, vec2 texCoords){

  // Bi-cubic texture sampling function, from https://jvm-gaming.org/index.php?topic=35123.0

	vec2 texSize = textureSize(sampler, 0);
	vec2 invTexSize = 1.0 / texSize;

	texCoords = texCoords * texSize - 0.5;


    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2(-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4(xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    vec4 sample0 = texture(sampler, offset.xz);
    vec4 sample1 = texture(sampler, offset.yz);
    vec4 sample2 = texture(sampler, offset.xw);
    vec4 sample3 = texture(sampler, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(
    	mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
    }

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
  }

vec4 mod289(vec4 x)
  {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
  }

vec4 permute(vec4 x)
  {
    return mod289(((x*34.0)+1.0)*x);
  }

vec4 taylorInvSqrt(vec4 r)
  {
    return 1.79284291400159 - 0.85373472095314 * r;
  }

vec2 fade(vec2 t) {
    return t*t*t*(t*(t*6.0-15.0)+10.0);
  }

float pnoise(vec2 P, vec2 rep, float seed){

    // Classic Perlin noise, periodic variant
    //
    // Author:  Stefan Gustavson (stefan.gustavson@liu.se)
    // Version: 2011-08-22
    //
    // Copyright (c) 2011 Stefan Gustavson. All rights reserved.
    // Distributed under the MIT license. See LICENSE file.
    // https://github.com/ashima/webgl-noise
    //

    vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
    vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
    Pi = mod(Pi, rep.xyxy); // To create noise with explicit period
    Pi = mod289(Pi);        // To avoid truncation effects in permutation
    vec4 ix = Pi.xzxz + seed;
    vec4 iy = Pi.yyww + seed;
    vec4 fx = Pf.xzxz;
    vec4 fy = Pf.yyww;

    vec4 i = permute(permute(ix) + iy);

    vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
    vec4 gy = abs(gx) - 0.5 ;
    vec4 tx = floor(gx + 0.5);
    gx = gx - tx;

    vec2 g00 = vec2(gx.x,gy.x);
    vec2 g10 = vec2(gx.y,gy.y);
    vec2 g01 = vec2(gx.z,gy.z);
    vec2 g11 = vec2(gx.w,gy.w);

    vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.w;

    float n00 = dot(g00, vec2(fx.x, fy.x));
    float n10 = dot(g10, vec2(fx.y, fy.y));
    float n01 = dot(g01, vec2(fx.z, fy.z));
    float n11 = dot(g11, vec2(fx.w, fy.w));

    vec2 fade_xy = fade(Pf.xy);
    vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
    float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
  }

float combined_pnoises(float a, float b, vec3 v0, vec3 v1, vec3 v2, float seed){
  vec2 vec_0 = vec2(v0.x*a,v0.y*b);
  vec2 vec_1 = vec2(v1.x*a,v1.y*b);
  vec2 vec_2 = vec2(v2.x*a,v2.y*b);
  vec2 per_0 = vec2(v0.x,v0.y);
  vec2 per_1 = vec2(v1.x,v1.y);
  vec2 per_2 = vec2(v2.x,v2.y);
  return v0.z*pnoise(vec_0,per_0,seed) + v1.z*pnoise(vec_1,per_1,seed) + v2.z*pnoise(vec_2,per_2,seed);
  }

float power_smooth_min( float a, float b, float k ){
    a = pow( a, k );
    b = pow( b, k );
    return pow( (a*b)/(a+b), 1.0/k );
  }

float power_smooth_min(float[20] A, float k){

    int n = A.length();

    for(int i=0; i<n; i++) {
      A[i] = pow(A[i],k);
    }

    float prod = 1.0;
    for(int i=0; i<n; i++) {
      prod = prod*A[i];
    }

    float sum = 0.0;
    for(int i=0; i<n; i++) {
      float prod2 = 1.0;
      for(int j=0; j<n; j++) {
        if(i==j){continue;}
        prod2 = prod2*A[j];
      }
      sum += prod2;
    }

    return pow(prod/sum, 1.0/k);
  }

float sum(float[20] V){
  int n = V.length();
  float sum_of_list = 0.0;
  for(int i=0; i<n; i++) {
    sum_of_list += V[i];
  }
  return sum_of_list;
  }

float min_all(float[20] A){
  int n = A.length();
  float min_of_list = 999999.9;
  for(int i=0; i<n; i++) {
    if (A[i]<min_of_list) {
      min_of_list = A[i];
    }
  }
  return min_of_list;
  }

//----------------------------------------------------------------------------------------------------------------------------------------------

void main()
{

    //Parameters for distance range of knot impact
    float range_k0 = 0.8;
    float range_k1 = 1.0;
    float range_d0 = 0.1;
    float range_d1 = 0.2;

    // GLOBAL 3D TEXTURE COORDINATES
    float Px = outTexCoords.x;
    float Py = outTexCoords.y + 0.18;                                 // offsetting texture coordinates in y-direction to make center of tree visible
    float Pz_p = outTexCoords.z + mod(0.25*time,zrat);            // animating z to show how the texture changes
    float Pz_m = map(Pz_p,0.0,zrat,0.0,1.0);                          // mapped to tree height, i.e. range 0.0-1.0

    // STEM GEOMETRY
    vec3 prm = vec3(textureBicubic(PithRadiusMap,vec2(0.5,Pz_m)));    // PAPER Section 4.4.1. Sampling pith_radius_map to get pith point offset at current height
    Px = Px-map(prm.x,0.0,1.0,-0.5,0.5);
    Py = Py-map(prm.y,0.0,1.0,-0.5,0.5);
    vec3 S_s = vec3(0.0,0.0,Pz_p);                                    // pseudo-closest point on stem skeleton
    vec3 P = vec3(Px,Py,Pz_p);
    float omega = map( mod( atan(Px,Py)+2*PI, 2*PI), 0, 2*PI, 0, 1.0);// rotation of P around vertical stem axis
    prm = vec3( textureBicubic(PithRadiusMap, vec2(omega,Pz_m) ) );   // PAPER Section 4.4.1. Sampling pith_radius_map to get radius (equivalent to speed of growth)
    float r_s = map(prm.b, 0.0, 1.0, 1.0, rmax/rmin);                 // local max radius of stem
    float d_s = distance(S_s,P);                                      // horizontal distance to pith
    float t_s = d_s/r_s;                                              // PAPER Equation 1. Calculate time value for stem

    // Define relevant distance range for knots
    float dist_range_0 = range_d0 + d_s*range_k0;
    float dist_range_1 = range_d1 + d_s*range_k1;

    // KNOT GEOMETRY
    // initiate knot variables
    int n = 20;         // max number of considred knots
    float D_b[20];      // distances to from P to branch skeleton points S_b
    float T_b[20];      // time values of knots
    float T_death[20];  // time of death values of knots
    float BETA[20];     // orientations of P around knot axes
    int IND[20];        // knot indices
    for(int i=0; i<n; i++) {
      D_b[i]=9.0;
      T_b[i]=9.0;
      T_death[i]=9.0;
      BETA[i]=0.0;
      IND[i]=0;
    }

    // Make list of up to n knots to be considred (knots within distance range)
    int cnt = 0;
    for(int i=0; i<N; i++) {

      // Sample knot maps. PAPER Section 4.4.2
      vec2 kmt = vec2( d_s, (i+0.5) / N );  //knots map texture coordinate. x: distance from S_s (stem center point), y: knot index ratio
      vec3 khm = vec3( textureBicubic( KnotHeightMap,   kmt ) );
      vec3 kom = vec3( textureBicubic( KnotOrientMap,   kmt ) );
      vec3 ksm = vec3( texture2D(      KnotStateMap,    kmt ) );

      // Get pseudo-nearest point (S_b) on branch skeleton
      float omega_b = map(kom.r,0.0,1.0,0.0,2*PI) + map(-kom.g+kom.b,0.0,1.0,0.0,0.5*PI);
      float bx = d_s*cos(omega_b);
      float by = d_s*sin(omega_b);
      float bz = map(khm.r, 0.0, 1.0, 0.0, zrat) - khm.g + khm.b;
      vec3 S_b = vec3(bx,by,bz);

      // Caclculate distance, and proceed if within range
      float d_b = distance(S_b,P);
      if (d_b<dist_range_1){ //the knot is within the upper bounds of the range

        D_b[cnt] = d_b;           //distance from texture point to pseudo-nearest branch skeleton point
        T_death[cnt] = ksm.g/r_s; //time knot died

        // Caclculate beta - angle around knot axis
        vec3 beta_vec = P-S_b;
        BETA[cnt] = mod( atan( beta_vec.z, beta_vec.x*sin(-omega_b)+beta_vec.y*cos(-omega_b) ) + 2*PI, 2*PI); //angle around knot axis

        // Create noise for knot radius r_b
        float beta_01 = map(BETA[cnt], 0.0, 2*PI, 0.0, 1.0);
        float seed = float(i)/N;
        float noise_value = combined_pnoises(beta_01, d_s, vec3(1.0, 1.0, 0.1), vec3(2.5, 3.0, 0.1), vec3(6.0, 7.0, 0.1), seed);
        float r_b = 0.2 - 0.1*sqrt(d_s) + 0.1*noise_value; //0.2 is an arbitrary parameter for the thickness/speed of growth of the knot
        T_b[cnt] = (d_b/r_b);
        cnt+=1;

        // Make smooth edge of distance range
        if (d_b>dist_range_0){
          float prog = (d_b - dist_range_0)/(dist_range_1-dist_range_0);
          T_b[cnt] = mix(T_b[cnt], 9.0, prog);
        }

        if (cnt>=n){ //max n knots considered at one point
          break;
          }
      }
    }

    // SMOOTH MERGE MINIMUM
    float t = 9.0;        //initiate combined time value
    float t_b_min = 9.0;  //initiate minimum of all knot time values

    //k-parameters of smoothness of min union
    float k_s = 1.5;
    float k_b = 5.0;

    //for-parameters of smoothness of min union with dead branches
    float f1 = -1.5;
    float f2 = 0.2;
    float f3 = 8.0;
    float f4 = 5.0; //speed of increasing k (decreasing smoothness) with time after death of knot

    //dead knot color parameters
    float dead_color_factor = 0;
    float dead_outline_factor = 1.0;
    float dead_outline_thickness = 0.02;

    float DELTA[20]; // list of delta - amount of smoothness

    for(int i=0; i<n; i++) {

      DELTA[i]=0;
      if (T_b[i]<9.0){

        // Calculate adaptive k-value. PAPER Section 4.2.1
        float t_Delta = t_s-T_b[i];
        float k = 0.5*(k_b-k_s)*t_Delta/(0.3+abs(t_Delta))+0.5*(k_b+k_s);

        // Smooth minimum. PAPER Equation 2
        t = power_smooth_min(t_s,T_b[i],k);

        // Amount of smoothing. PAPER Equation 3
        DELTA[i] = t - min(t_s,T_b[i]);

        // If after knot died. Paper Section 4.2.2
        if (t>T_death[i]) {

          float t_after_death = abs(t_s-T_death[i]);

          // Stop expansion in thickness. Paper Section 4.2.2
          T_b[i] += t_after_death;
          float t = power_smooth_min(t_s,T_b[i],k+f4*t_after_death);
          float delta =  t - min(t_s,T_b[i]);

          // Reduce and invert smoothing. Paper Section 4.2.2
          float td2 = f3*t_after_death-1.0;
          float f_range = f2-f1;
          float f = 1.0 - 0.5*f_range*( td2/(0.3+abs(td2)) ) + f1 + 0.5*f_range ;
          DELTA[i] = f*delta;

        }

        // Transition at boarder of distance range
        if (D_b[i]>dist_range_0 && D_b[i]<=dist_range_1) {
          float prog = D_b[i]-dist_range_0;
          float range = dist_range_1-dist_range_0;
          prog = prog/range;
          DELTA[i] -= DELTA[i]*prog;
        }

        // Inside dead knot (variable for coloring)
        if (T_b[i]<T_death[i] && t_s>T_death[i]){
            dead_color_factor = t_s-T_death[i];
        }

        if (T_b[i]<T_death[i]){ // Dead knot outline
          float beta_01 = map(BETA[i],0.0,2*PI,0.0,1.0);
          float seed = float(IND[i])/N;
          float noise_value = combined_pnoises(beta_01, d_s, vec3(2.0, 1.0, 0.005), vec3(5.0, 2.0, 0.008), vec3(23.0, 5.0, 0.010), seed);
          dead_outline_thickness+=noise_value;
          if( abs(T_death[i]-t)<dead_outline_thickness){
            dead_outline_factor = 0.65;
          }
        }
      }
    }

    t_b_min = power_smooth_min(T_b, 2.0);

    // PUTTING IT TOGETHER
    // Make sure a knot does not effet the interior of another knot
    for(int i=0; i<n; i++) {
      float delta = max(t_s-t_b_min,0.0)/t_s;
      delta = smoothstep(0.0,1.0,delta);
      DELTA[i] -= DELTA[i]*delta;
    }
    // Apply smoothing (min + delta). PAPER Equation 6
    float delta_sum = sum(DELTA);
    float t_min = min(t_s, min_all(T_b));
    t = t_min + delta_sum;

    // COLOR. PAPER Section 4.3
    vec3 texColor = vec3(texture2D(ColorMap, vec2(t,0.5)));
    vec3 knotColor = vec3(0.20,0.20,0.15); //arbitrary color
    float m = 14;
    float g = 1/pow(clamp(1.2*t_b_min-t_s,0.001,1.0)+1.0,m);
    texColor -= g*knotColor; //darken knot (alive and dead)
    texColor -= g*clamp(3*dead_color_factor, 0.0, 0.5)*knotColor; //furhter darken dead knot
    texColor = dead_outline_factor*texColor; //outline of dead knot

    // Apply normal map
    vec3 normal_rgb = vec3(texture2D(NormalMap, vec2(t,0.5)));
    float gg = 2 * clamp ( t_s - t_b_min + 0.3, 0.0, 1.0) ;
    float xlen = (1.0-gg) * ( 2.0*normal_rgb.x-1.0 );
    float zlen = abs(2*normal_rgb.z-1.0);
    vec3 Nv = normalize(normal);
    vec3 Tv = normalize(vec3(P.x,P.y,0.0));
    vec3 Bv = cross(Nv,Tv);
    Tv = cross(Bv,Nv);
    vec3 norm = normalize(xlen*Tv + zlen*Nv);

    // LIGHT (Phong lighting model)
    vec3 lightColor = vec3(1.0,1.0,0.95);
    // Ambient
    float ambientStrength = 0.8;
    vec3 ambient = ambientStrength * lightColor;
    // Diffuse
    float roughness = 0.5;
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = roughness * diff * lightColor;
    //Specular
    float specularStrength = 0.4;
    float shininess = vec3(texture2D(SpecularMap, vec2(t,0.5))).r;
    vec3 viewDir = normalize(viewPos-fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0),32);
    vec3 specular = shininess * specularStrength * spec * lightColor;
    //Combine
    vec3 result = (ambient + diffuse + specular) * texColor;
    outColor = vec4(result, 1.0);
}

// TO DOs
// To do : Gradual inversion around dead knot - code could be more readable and intuative to edit
// To do : Knot coloring (alive, dead, dead outline) - code could be more readable and intuative to edit
