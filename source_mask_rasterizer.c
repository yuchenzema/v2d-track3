#include <math.h>
#include <stdint.h>
/* Union of projected triangles. Overlapping front/back faces never cancel. */
void silhouette(const double *p, const int32_t *f, int count, int width, int height, uint8_t *mask) {
    for (int i=0; i<count; ++i) {
        const double *a=p+3*f[3*i], *b=p+3*f[3*i+1], *c=p+3*f[3*i+2];
        if (a[2]<=.05 || b[2]<=.05 || c[2]<=.05) continue;
        double minx=fmin(a[0],fmin(b[0],c[0])), maxx=fmax(a[0],fmax(b[0],c[0]));
        double miny=fmin(a[1],fmin(b[1],c[1])), maxy=fmax(a[1],fmax(b[1],c[1]));
        if (!isfinite(minx) || !isfinite(maxx) || !isfinite(miny) || !isfinite(maxy)) continue;
        if (maxx<0 || minx>=width || maxy<0 || miny>=height) continue;
        int x0=(int)floor(fmax(0,minx)), x1=(int)ceil(fmin(width-1,maxx));
        int y0=(int)floor(fmax(0,miny)), y1=(int)ceil(fmin(height-1,maxy));
        double den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
        if (fabs(den)<1e-12) continue;
        for (int y=y0; y<=y1; ++y) for (int x=x0; x<=x1; ++x) {
            if (mask[y*width+x]) continue;
            double u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/den;
            double v=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/den;
            if (u>=-1e-9 && v>=-1e-9 && 1-u-v>=-1e-9) mask[y*width+x]=1;
        }
    }
}

/* Perspective-correct front surface depth in metres, zero for background. */
void surface_depth(const double *p, const int32_t *f, int count, int width, int height, float *depth) {
    for (int i=0; i<count; ++i) {
        const double *a=p+3*f[3*i], *b=p+3*f[3*i+1], *c=p+3*f[3*i+2];
        if (a[2]<=.05 || b[2]<=.05 || c[2]<=.05) continue;
        double minx=fmin(a[0],fmin(b[0],c[0])), maxx=fmax(a[0],fmax(b[0],c[0]));
        double miny=fmin(a[1],fmin(b[1],c[1])), maxy=fmax(a[1],fmax(b[1],c[1]));
        if (!isfinite(minx) || !isfinite(maxx) || !isfinite(miny) || !isfinite(maxy)) continue;
        if (maxx<0 || minx>=width || maxy<0 || miny>=height) continue;
        int x0=(int)floor(fmax(0,minx)), x1=(int)ceil(fmin(width-1,maxx));
        int y0=(int)floor(fmax(0,miny)), y1=(int)ceil(fmin(height-1,maxy));
        double den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
        if (fabs(den)<1e-12) continue;
        for (int y=y0; y<=y1; ++y) for (int x=x0; x<=x1; ++x) {
            double u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/den;
            double v=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/den;
            double w=1.-u-v;
            if (u>=-1e-9 && v>=-1e-9 && w>=-1e-9) {
                float z=(float)(1./(u/a[2]+v/b[2]+w/c[2]));
                int pixel=y*width+x;
                if (depth[pixel]==0.f || z<depth[pixel]) depth[pixel]=z;
            }
        }
    }
}
