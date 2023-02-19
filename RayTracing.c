#define _USE_MATH_DEFINES

// Uncomment these if you have installed the libraries
//#define USE_OMP
//#define USE_SDL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_SDL
#include "SDL.h"
#include "SDL_keyboard.h"
#endif

#define GridSize 1001
#define MaxDepth 5

#define vacuum 0
#define light 1
#define diffuseRed 2
#define diffuseGreen 3
#define diffuseBlue 4
#define diffuseWhite 5
#define reflectiveBlue 6
#define reflectiveRed 7

#define false 0
#define true 1

typedef unsigned char byte;
typedef char bool;

struct Vector3
{
	double x, y, z;
};
typedef struct Vector3 Vector3;

struct Vector2
{
	double x, y;
};
typedef struct Vector2 Vector2;

struct Color
{
	byte r, g, b;
};
typedef struct Color Color;

struct Camera
{
	Vector3 position;
	Vector3 orientation;
	double focalLength, pixelSize; // in mm
	Vector2 resolution;
	Vector3** imagePoints; //[0][0] top left
	uint32_t* pixels;
};
typedef struct Camera Camera;

struct Ray
{
	Vector3 o; // origin
	Vector3 w; // direction
};
typedef struct Ray Ray;

struct Material
{
	Vector3 color;
	double emittance;
	double reflectance;
	double kd;
	double ks;
	double shininess;
};
typedef struct Material Material;

enum RenderType
{
	Depth = 0,
	RayCasting = 1,
	RayTracing = 2,
	PathTracing = 3,
	Radiosity = 4
};
typedef enum RenderType RenderType;
char RenderTypeToString[][20] = { "Depth", "RayCasting", "RayTracing", "PathTracing", "Radiosity" };

Vector2 NewVector2(int x, int y)
{
	Vector2 v;
	v.x = x;
	v.y = y;
	return v;
}

Vector3 NewVector3(double x, double y, double z)
{
	Vector3 v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

Color NewColor(byte r, byte g, byte b)
{
	Color color;
	color.r = r;
	color.g = g;
	color.b = b;
	return color;
}

Ray NewRay(Vector3 origin, Vector3 direction)
{
	Ray ray;
	ray.o = origin;
	ray.w = direction;
	return ray;
}

Color ColorAdd(Color c1, Color c2)
{
	Color r;
	r.r = c1.r + c2.r;
	r.g = c1.g + c2.g;
	r.b = c1.b + c2.b;
	return r;
}

Color ColorDivScalar(Color c, double s)
{
	Color r;
	r.r = c.r / s;
	r.g = c.g / s;
	r.b = c.b / s;
	return r;
}

Color ColorMulScalar(double s, Color c)
{
	Color r;
	r.r = s * c.r;
	r.g = s * c.g;
	r.b = s * c.b;
	return r;
}

char* Vector3ToString(Vector3 v)
{
	static char buffer[100];
	sprintf(buffer, "(%lf,%lf,%lf)", v.x, v.y, v.z);
	return buffer;
}

char* RayToString(Ray ray)
{
	static char buffer[100];
	sprintf(buffer, "o = (%lf,%lf,%lf), w = (%lf,%lf,%lf)", ray.o.x, ray.o.y, ray.o.z, ray.w.x, ray.w.y, ray.w.z);
	return buffer;
}

Vector3 add(Vector3 v1, Vector3 v2)
{
	Vector3 r;
	r.x = v1.x + v2.x;
	r.y = v1.y + v2.y;
	r.z = v1.z + v2.z;
	return r;
}

Vector3 sub(Vector3 v1, Vector3 v2)
{
	Vector3 r;
	r.x = v1.x - v2.x;
	r.y = v1.y - v2.y;
	r.z = v1.z - v2.z;
	return r;
}

Vector3 mul(Vector3 v1, Vector3 v2)
{
	Vector3 r;
	r.x = v1.x * v2.x;
	r.y = v1.y * v2.y;
	r.z = v1.z * v2.z;
	return r;
}

Vector3 vecdiv(Vector3 v1, Vector3 v2)
{
	Vector3 r;
	r.x = v2.x == 0 ? v1.x : v1.x / v2.x;
	r.y = v2.y == 0 ? v1.y : v1.y / v2.y;
	r.z = v2.z == 0 ? v1.z : v1.z / v2.z;
	return r;
}

Vector3 vecpow(Vector3 v, Vector3 p)
{
	Vector3 r;
	r.x = pow(v.x, p.x);
	r.y = pow(v.y, p.y);
	r.z = pow(v.z, p.z);
	return r;
}

Vector3 mul_s(double scalar, Vector3 v)
{
	Vector3 r;
	r.x = scalar * v.x;
	r.y = scalar * v.y;
	r.z = scalar * v.z;
	return r;
}

Vector3 div_s(Vector3 v, double scalar)
{
	Vector3 r;
	if (scalar == 0)
		return v;
	r.x = v.x / scalar;
	r.y = v.y / scalar;
	r.z = v.z / scalar;
	return r;
}

double norm(Vector3 v)
{
	double norm = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
	return norm;
}

Vector3 normalize(Vector3 v)
{
	return div_s(v, norm(v));
}

double dist(Vector3 v1, Vector3 v2)
{
	return norm(sub(v1, v2));
}

double dot(Vector3 v1, Vector3 v2)
{
	double r = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	return r;
}

Vector3 cross(Vector3 v1, Vector3 v2)
{
	Vector3 r;
	r.x = v1.y * v2.z - v1.z * v2.y;
	r.y = v1.z * v2.x - v1.x * v2.z;
	r.z = v1.x * v2.y - v1.y * v2.x;
	return r;
}

Vector3 translate(Vector3 point, Vector3 translation)
{
	point.x = point.x + translation.x;
	point.y = point.y + translation.y;
	point.z = point.z + translation.z;
	return point;
}

Vector3 rotate(Vector3 point, Vector3 orientation)
{
	Vector3 r;
	double pitch = orientation.x * M_PI / 180;
	double yaw = orientation.y * M_PI / 180;
	double roll = orientation.z * M_PI / 180;

	double cosa = cos(roll);
	double sina = sin(roll);

	double cosb = cos(yaw);
	double sinb = sin(yaw);

	double cosc = cos(pitch);
	double sinc = sin(pitch);

	double Axx = cosa * cosb;
	double Axy = cosa * sinb * sinc - sina * cosc;
	double Axz = cosa * sinb * cosc + sina * sinc;

	double Ayx = sina * cosb;
	double Ayy = sina * sinb * sinc + cosa * cosc;
	double Ayz = sina * sinb * cosc - cosa * sinc;

	double Azx = -sinb;
	double Azy = cosb * sinc;
	double Azz = cosb * cosc;

	r.x = Axx * point.x + Axy * point.y + Axz * point.z;
	r.y = Ayx * point.x + Ayy * point.y + Ayz * point.z;
	r.z = Azx * point.x + Azy * point.y + Azz * point.z;

	return r;
}

Vector3 scale(Vector3 point, Vector3 scale)
{
	point.x = point.x * scale.x;
	point.y = point.y * scale.y;
	point.z = point.z * scale.z;
	return point;
}

Vector3 reflect(Vector3 v, Vector3 n)
{
	Vector3 r = sub(v, mul_s(2 * dot(v, n), n));
	return r;
}

bool range(double x, double r1, double r2)
{
	return x >= r1 && x <= r2;
}

double randomRange(double min, double max)
{
	double r = min + (max - min) * (rand() / (double)(RAND_MAX));
	return r;
}

Vector3 clamp(Vector3 v, Vector3 min, Vector3 max)
{
	if (v.x < min.x)
		v.x = min.x;
	if (v.x > max.x)
		v.x = max.x;
	if (v.y < min.y)
		v.y = min.y;
	if (v.y > max.y)
		v.y = max.y;
	if (v.z < min.z)
		v.z = min.z;
	if (v.z > max.z)
		v.z = max.z;
	return v;
}

double Vector3MaxElement(Vector3 v)
{
	if (v.x >= v.y && v.x >= v.z)
		return v.x;
	if (v.y >= v.x && v.y >= v.z)
		return v.y;
	else //if (v.z >= v.x && v.z >= v.y)
		return v.z;
}

byte*** voxelGrid; // 1 cubic meter voxel grid
Camera* camera;
Material* materials;
Vector3 lightPos; // light position for phong lighting

void CreateGrid()
{
	int i, j, k;
	voxelGrid = malloc(GridSize * sizeof(byte**));
	for (i = 0; i < GridSize; i++)
	{
		voxelGrid[i] = malloc(GridSize * sizeof(byte*));
		for (j = 0; j < GridSize; j++)
		{
			voxelGrid[i][j] = malloc(GridSize * sizeof(byte));
			for (k = 0; k < GridSize; k++)
			{
				voxelGrid[i][j][k] = 0;
			}
		}
	}
}

void DeleteGrid()
{
	int i, j;
	for (i = 0; i < GridSize; i++)
	{
		for (j = 0; j < GridSize; j++)
		{
			free(voxelGrid[i][j]);
		}
		free(voxelGrid[i]);
	}
	free(voxelGrid);
}

void CreateMaterials()
{
	materials = malloc(10 * sizeof(Material));

	// vacuum
	materials[0].color = NewVector3(0, 0, 0); 
	materials[0].emittance = 0;
	materials[0].reflectance = 0;
	materials[0].kd = 0;
	materials[0].ks = 0;
	materials[0].shininess = 0;

	// light
	materials[1].color = NewVector3(1.5, 1.5, 1.5);
	materials[1].emittance = 1;
	materials[1].reflectance = 0;
	materials[1].kd = 0;
	materials[1].ks = 0;
	materials[1].shininess = 10;

	// diffuse red
	materials[2].color = NewVector3(1, 0, 0);
	materials[2].emittance = 0;
	materials[2].reflectance = 0;
	materials[2].kd = 0.9;
	materials[2].ks = 0.5;
	materials[2].shininess = 10;

	// diffuse green
	materials[3].color = NewVector3(0, 1, 0);
	materials[3].emittance = 0;
	materials[3].reflectance = 0;
	materials[3].kd = 0.9;
	materials[3].ks = 0.5;
	materials[3].shininess = 10;

	// diffuse blue
	materials[4].color = NewVector3(0, 0, 1);
	materials[4].emittance = 0;
	materials[4].reflectance = 0;
	materials[4].kd = 0.9;
	materials[4].ks = 0.5;
	materials[4].shininess = 10;

	// diffuse white
	materials[5].color = NewVector3(1, 1, 1);
	materials[5].emittance = 0;
	materials[5].reflectance = 0;
	materials[5].kd = 0.9;
	materials[5].ks = 0.5;
	materials[5].shininess = 10;

	// reflective blue
	materials[6].color = NewVector3(0, 0, 1);
	materials[6].emittance = 0;
	materials[6].reflectance = 0.1;
	materials[6].kd = 0.8;
	materials[6].ks = 0.5;
	materials[6].shininess = 10;

	// reflective red
	materials[7].color = NewVector3(1, 0, 0);
	materials[7].emittance = 0;
	materials[7].reflectance = 0.1;
	materials[7].kd = 0.8;
	materials[7].ks = 0.5;
	materials[7].shininess = 10;
}

void CreateCube(Vector3 center, double size, byte material)
{
	int x1 = (center.x - size/2) * (GridSize - 1);
	int x2 = (center.x + size/2) * (GridSize - 1);
	int y1 = (center.y - size/2) * (GridSize - 1);
	int y2 = (center.y + size/2) * (GridSize - 1);
	int z1 = (center.z - size/2) * (GridSize - 1);
	int z2 = (center.z + size/2) * (GridSize - 1);
	int i, j, k;
	for (i = x1; i < x2; i++)
	{
		for (j = y1; j < y2; j++)
		{
			for (k = z1; k < z2; k++)
			{
				voxelGrid[i][j][k] = material;
			}
		}
	}
}

void CreateSphere(Vector3 center, double radius, byte material)
{
	int x1 = (center.x - radius) * (GridSize - 1);
	int x2 = (center.x + radius) * (GridSize - 1);
	int y1 = (center.y - radius) * (GridSize - 1);
	int y2 = (center.y + radius) * (GridSize - 1);
	int z1 = (center.z - radius) * (GridSize - 1);
	int z2 = (center.z + radius) * (GridSize - 1);
	int i, j, k;
	Vector3 point;
	double distance;
	center = mul_s(GridSize, center);
	radius = radius * GridSize;
	for (i = x1; i < x2; i++)
	{
		for (j = y1; j < y2; j++)
		{
			for (k = z1; k < z2; k++)
			{
				point = NewVector3(i, j, k);
				distance = norm(sub(point, center));
				if (distance <= radius)
					voxelGrid[i][j][k] = material;
			}
		}
	}
}

void CreateLight(Vector3 lightPosition)
{
	voxelGrid[(int)lightPosition.x * (GridSize - 1)][(int)lightPosition.y * (GridSize - 1)][(int)lightPosition.z * (GridSize - 1)] = light;
	CreateCube(lightPosition, 0.02, light);
	//CreateSphere(lightPos, 0.01, light);
	lightPos = lightPosition; // Set global variable for easy access
}

void CreateWalls()
{
	int i, j;
	for (i = 0; i < GridSize; i++)
	{
		for (j = 0; j < GridSize; j++)
		{
			voxelGrid[i][j][0] = diffuseWhite;
			voxelGrid[i][j][GridSize-1] = diffuseWhite;

			//voxelGrid[i][0][j] = material; // floor
			voxelGrid[i][GridSize-1][j] = diffuseWhite;

			voxelGrid[0][i][j] = diffuseRed;
			voxelGrid[GridSize-1][i][j] = diffuseGreen;
		}
	}
}

void CreateFloor()
{
	int i, j;
	for (i = 0; i < GridSize; i++)
	{
		for (j = 0; j < GridSize; j++)
		{
			voxelGrid[i][0][j] = diffuseWhite;
		}
	}
}

void DeleteImagePoints()
{
	int i;
	for (i = 0; i < camera->resolution.y; i++)
	{
		free(camera->imagePoints[i]);
	}
	free(camera->imagePoints);
}

void DeleteCamera()
{
	DeleteImagePoints();
	free(camera);
}

void ApplyTransformationsToCameraPoints()
{
	int i, j;
	Vector3 cameraForward = normalize(rotate(NewVector3(0, 0, 1), camera->orientation));
	for (i = 0; i < camera->resolution.y; i++)
	{
		for (j = 0; j < camera->resolution.x; j++)
		{
			camera->imagePoints[i][j].x = j - (camera->resolution.x - 1.0) / 2;
			camera->imagePoints[i][j].y = -i + (camera->resolution.y - 1.0) / 2;
			camera->imagePoints[i][j].z = camera->focalLength;

			// Apply object to world space camera transformations (translation, rotation and scale)
			camera->imagePoints[i][j] = scale(camera->imagePoints[i][j], NewVector3(camera->pixelSize/1000, camera->pixelSize/1000, 1.0/1000));
			camera->imagePoints[i][j] = rotate(camera->imagePoints[i][j], camera->orientation);
			camera->imagePoints[i][j] = translate(camera->imagePoints[i][j], camera->position);
		}
	}
}

void CreateImagePoints()
{
	int i;
	camera->imagePoints = malloc(camera->resolution.y * sizeof(Vector3*));
	for (i = 0; i < camera->resolution.y; i++)
		camera->imagePoints[i] = malloc(camera->resolution.x * sizeof(Vector3));
	ApplyTransformationsToCameraPoints();
}

void CreateCamera(Vector3 pos, Vector3 orientation, double focalLength, double pixelSize, Vector2 resolution)
{
	camera = malloc(sizeof(Camera));
	camera->position = pos;
	camera->orientation = orientation;
	camera->focalLength = focalLength;
	camera->pixelSize = pixelSize;
	camera->resolution = resolution;
	CreateImagePoints();
	camera->pixels = malloc(camera->resolution.x * camera->resolution.y * sizeof(uint32_t));
}

void UpdateCameraParameters(Vector3 pos, Vector3 orientation, double focalLength)
{
	camera->position = pos;
	camera->orientation = orientation;
	camera->focalLength = focalLength;
	ApplyTransformationsToCameraPoints();
}

// Closest volume point to v
Vector3 ClosestVolumePoint(Vector3 v)
{
	return NewVector3(round(v.x * (GridSize - 1)), round(v.y * (GridSize - 1)), round(v.z * (GridSize - 1)));
}

byte SampleVolume(Vector3 v)
{
	return voxelGrid[(int)round(v.x * (GridSize-1))][(int)round(v.y * (GridSize-1))][(int)round(v.z * (GridSize-1))];
}

Vector3 Normal(Vector3 v)
{
	int x = (int)round(v.x*(GridSize-1));
	int y = (int)round(v.y*(GridSize-1));
	int z = (int)round(v.z*(GridSize-1));
	Vector3 p = NewVector3(x, y, z); // ClosestVolumePoint(v)
	Vector3 normal;
	byte material = voxelGrid[x][y][z];

	// Check if we are at the edge
	if (x == GridSize - 1)
		return NewVector3(-1, 0, 0);
	if (x == 0)
		return NewVector3(1, 0, 0);
	if (y == GridSize - 1)
		return NewVector3(0, -1, 0);
	if (y == 0)
		return NewVector3(0, 1, 0);
	if (z == GridSize - 1)
		return NewVector3(0, 0, -1);
	if (z == 0)
		return NewVector3(0, 0, 1);

	// Compute centroid of cube volume around the point
	int cubeSize = 30;
	Vector3 sum = NewVector3(0, 0, 0);
	int count = 0;
	Vector3 point;
	Vector3 centroid;
	int i, j, k;
	for (i = x - cubeSize / 2; i < x + cubeSize / 2 - 1; i++)
	{
		for (j = y - cubeSize / 2; j < y + cubeSize / 2 - 1; j++)
		{
			for (k = z - cubeSize / 2; k < z + cubeSize / 2 - 1; k++)
			{
				if (!range(i, 0, GridSize - 1) || !range(j, 0, GridSize - 1) || !range(k, 0, GridSize - 1))
					continue;
				if (voxelGrid[i][j][k] == material)
				{
					point = NewVector3(i, j, k);
					sum = add(sum, point);
					count++;
				}
			}
		}
	}
	centroid = div_s(sum, count);
	normal = sub(p, centroid);

	return normal;
}

Vector3 Intersect(Ray ray)
{
	double step = 0.001;
	Vector3 rayContact = add(ray.o, mul_s(step, ray.w));
	byte material;
	byte startingMaterial = voxelGrid[(int)round(ray.o.x * (GridSize - 1))][(int)round(ray.o.y * (GridSize - 1))][(int)round(ray.o.z * (GridSize - 1))];
	while (true)
	{
		material = voxelGrid[(int)round(rayContact.x * (GridSize - 1))][(int)round(rayContact.y * (GridSize - 1))][(int)round(rayContact.z * (GridSize - 1))];
		if (material != startingMaterial && material != vacuum) // found material
			return rayContact;
		rayContact = add(rayContact, mul_s(step, ray.w)); // ray.o + step * ray.w
	}
	return NewVector3(-1, -1, -1);
}

Vector3 ComputeDepth(Ray ray)
{
	Vector3 intersectionPoint;
	Vector3 I;
	double distance, maxDistance, d;

	maxDistance = sqrt(3); // Vector3Norm(NewVector3(1, 1, 1));
	intersectionPoint = Intersect(ray);
	distance = norm(sub(intersectionPoint, camera->position)); // ||intersectionPoint - camera->position||
	//distance = fabs(intersectionPoint.z - camera->position.z); // Alternatively for speedup (for orientation (0,0,0))
	d = 1 - distance / maxDistance;
	I = NewVector3(d, d, d);
	return I;
}

Vector3 Cast(Ray ray)
{
	Vector3 intersectionPoint, normal, cameraPos, lightDir, viewDir, reflectDir, I, lightColor;
	double ambient, diffuse, specular, cos_theta, I_i, I_max;
	byte material_id;
	Material material;

	intersectionPoint = Intersect(ray);
	if (intersectionPoint.x == -1) // nothing was hit
		return NewVector3(0, 0, 0);
	normal = Normal(intersectionPoint);
	material_id = SampleVolume(intersectionPoint);
	material = materials[material_id];
	if (material_id == light)
		return NewVector3(1, 1, 1);

	// Phong Lighting
	ambient = 0.1;
	normal = normalize(normal);
	lightDir = normalize(sub(lightPos, intersectionPoint));
	I_i = (SampleVolume(Intersect(NewRay(intersectionPoint, lightDir))) == light) * 1.0; // compute incoming light for shadow
	lightColor = NewVector3(I_i, I_i, I_i);
	diffuse = max(dot(normal, lightDir), 0.0) * 0.9;
	viewDir = normalize(sub(camera->position, intersectionPoint));
	reflectDir = normalize(mul_s(-1, reflect(lightDir, normal)));
	specular = pow(max(dot(viewDir, reflectDir), 0.0), 10) * 0.5;
	I = add(add(mul_s(ambient, material.color), mul_s(diffuse, mul(material.color, lightColor))), mul_s(specular, lightColor));

	return I;
}

Vector3 Trace(Ray ray, int depth)
{
	Vector3 intersectionPoint, normal, cameraPos, lightDir, viewDir, reflectDir, I, lightColor, I_r;
	double ambient, diffuse, specular, cos_theta, I_i;
	byte material_id, shadow_material;
	Material material;

	if (depth >= MaxDepth)
		return NewVector3(0, 0, 0);

	intersectionPoint = Intersect(ray);
	if (intersectionPoint.x == -1) // nothing was hit
		return NewVector3(0, 0, 0);
	normal = Normal(intersectionPoint);
	material_id = SampleVolume(intersectionPoint);
	material = materials[material_id];
	if (material_id == light)
		return NewVector3(1.5, 1.5, 1.5);

	// Phong Lighting
	ambient = 0.1;
	normal = normalize(normal);
	lightDir = normalize(sub(lightPos, intersectionPoint));
	I_i = (SampleVolume(Intersect(NewRay(intersectionPoint, lightDir))) == light) * 1.0; // compute incoming light for shadow
	lightColor = NewVector3(I_i, I_i, I_i);
	diffuse = max(dot(normal, lightDir), 0.0) * material.kd;
	viewDir = normalize(sub(camera->position, intersectionPoint));
	reflectDir = normalize(mul_s(-1, reflect(lightDir, normal)));
	specular = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess) * material.ks;

	// Recursion
	Ray newRay;
	newRay.o = intersectionPoint;
	newRay.w = normalize(reflect(ray.w, normal));
	if (material.reflectance > 0)
		I_r = Trace(newRay, depth + 1);
	else
		I_r = NewVector3(0, 0, 0);

	I = add(add(add(mul_s(ambient, material.color), mul_s(diffuse, mul(material.color, lightColor))), mul_s(specular, lightColor)), mul_s(material.reflectance, I_r));

	return I;
}

void SetPixelColor(int k, Color color)
{
	((unsigned char*)camera->pixels)[k * 4 + 0] = color.b;
	((unsigned char*)camera->pixels)[k * 4 + 1] = color.g;
	((unsigned char*)camera->pixels)[k * 4 + 2] = color.r;
	((unsigned char*)camera->pixels)[k * 4 + 3] = 255;
}

Color ToneMapping(Vector3 I)
{
	Color color;
	double I_max = 1.6;
	double gamma = 1.0 / 2.2;
	
	I = div_s(I, I_max); // tone mapping

	I = vecpow(I, NewVector3(gamma, gamma, gamma)); // gamma correction

	I = clamp(I, NewVector3(0, 0, 0), NewVector3(1, 1, 1));
	color.r = (int)(I.x * 255);
	color.g = (int)(I.y * 255);
	color.b = (int)(I.z * 255);
	return color;
}

void Render(RenderType type)
{
	int i, j, k, s;
	Vector3 imagePoint;
	Vector3 I;
	Ray ray;
	Color pixelColor;
	int pixels = camera->resolution.y * camera->resolution.x;

#ifdef USE_OMP
	#pragma omp parallel for private(i, j, imagePoint, ray, I, pixelColor)
#endif
	for (k = 0; k < pixels; k++)
	{
		i = k / (int)camera->resolution.y;
		j = k % (int)camera->resolution.y;
		imagePoint = camera->imagePoints[i][j];

		// Initialize ray from camera pos and image point and compute pixel value
		ray.o = imagePoint;
		ray.w = normalize(sub(imagePoint, camera->position));
		if (type == Depth)
			I = ComputeDepth(ray);
		else if (type == RayCasting)
			I = Cast(ray);
		else if (type == RayTracing)
			I = Trace(ray, 0);
		pixelColor = ToneMapping(I);
		SetPixelColor(k, pixelColor);
	}
}

void SaveImageToFile(uint32_t* pixels, Vector2 res, char* filename)
{
	int k;
	int pixelCount = res.x * res.y;
	FILE* file = fopen(filename, "w");
	fprintf(file, "P3\n%d %d\n%d\n", (int)res.x, (int)res.y, 255); // ppm file header
	for (k = 0; k < pixelCount; k++)
		fprintf(file, "%d %d %d\n", ((byte*)pixels)[k*4 + 2], ((byte*)pixels)[k*4 + 1], ((byte*)pixels)[k*4 + 0]);
	fclose(file);
}

Vector3 OrientationToDirectionVector(Vector3 orientation)
{
	Vector3 forward = NewVector3(0, 0, 1);
	Vector3 direction = rotate(forward, orientation);
	direction = normalize(direction);
	return direction;
}

#ifdef USE_SDL
void SDL_WaitKey() // Wait for key press
{
	SDL_Event event;
	while (!(SDL_WaitEvent(&event) && event.type == SDL_KEYDOWN));
}
#endif

#ifdef USE_SDL
void CloseWindowAndQuit(SDL_Window* sdlWindow)
{
	SDL_DestroyWindow(sdlWindow);
	SDL_Quit();
	DeleteGrid();
	DeleteCamera();
}
#endif

int main(int argv, char** args)
{
	// Options
	Vector3 lightPosition = NewVector3(0.46, 0.54, 0.27);
	Vector3 cubePos = NewVector3(0.6, 0.6, 0.5);
	Vector3 spherePos = NewVector3(0.4, 0.4, 0.5);
	Vector3 cameraPos = NewVector3(0.5, 0.5, 0.1);
	Vector3 cameraOrientation = NewVector3(0, 0, 0);
	double focalLength = 70; // in mm
	double pixelSize = 0.1; // in mm
	Vector2 cameraRes = NewVector2(1000, 1000); // Configurations: [0.1 1000 1000] [1 100 100] [2 50 50] [4 25 25]
	RenderType type = RayTracing;
#ifdef USE_SDL
	bool fullscreen = false;
	bool interactive = false;
	bool mouseRotation = false;
	double movementSpeed = 0.01;
	double rotationSpeed = 0.1;
#endif

	Vector3 forward = NewVector3(0, 0, 1);
	Vector3 right = NewVector3(1, 0, 0);
	Vector3 up = NewVector3(0, 1, 0);

#ifdef USE_SDL
	SDL_Window* sdlWindow = NULL;
	SDL_Renderer* sdlRenderer = NULL;
	SDL_Texture* sdlTexture = NULL;
	uint32_t sdlFlag;
	uint8_t* keyboard;
	uint32_t mouseButtons;
	int mouseDeltaX;
	int mouseDeltaY;
#endif

	clock_t t;
	double delta_time;
	char title[200];
	Vector3 cameraForward;
	Vector3 cameraRight;
	Vector3 cameraUp;

	// Create Grid
	CreateGrid();
	CreateMaterials();

	// Create Scene
	CreateLight(lightPosition);
	CreateWalls();
	CreateFloor();
	CreateCube(cubePos, 0.1, diffuseBlue);
	CreateSphere(spherePos, 0.1, reflectiveRed);
	CreateCamera(cameraPos, cameraOrientation, focalLength, pixelSize, cameraRes);
	
#ifdef USE_SDL
	// Init display
	if (interactive)
	{
		sdlFlag = fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : SDL_WINDOW_RESIZABLE;
		SDL_CreateWindowAndRenderer(700, 700, sdlFlag, &sdlWindow, &sdlRenderer);
		SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear"); // make the scaled rendering look smoother.
		SDL_RenderSetLogicalSize(sdlRenderer, camera->resolution.x, camera->resolution.y);
		sdlTexture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, camera->resolution.x, camera->resolution.y);
		SDL_SetRelativeMouseMode(mouseRotation);
	}

	// Main loop
	while (interactive)
	{
		t = clock();
		
		// Read input
		SDL_PumpEvents();
		keyboard = SDL_GetKeyboardState(NULL);
		mouseButtons = SDL_GetRelativeMouseState(&mouseDeltaX, &mouseDeltaY);

		if (keyboard[SDL_SCANCODE_ESCAPE])
			CloseWindowAndQuit(sdlWindow);
		if (keyboard[SDL_SCANCODE_1])
			type = Depth;
		if (keyboard[SDL_SCANCODE_2])
			type = RayCasting;
		if (keyboard[SDL_SCANCODE_3])
			type = RayTracing;
		if (keyboard[SDL_SCANCODE_4])
			type = PathTracing;
		if (keyboard[SDL_SCANCODE_5])
			type = Radiosity;
		if (keyboard[SDL_SCANCODE_UP])
			cameraOrientation = add(cameraOrientation, NewVector3(-50*rotationSpeed, 0, 0));
		if (keyboard[SDL_SCANCODE_DOWN])
			cameraOrientation = add(cameraOrientation, NewVector3(50*rotationSpeed, 0, 0));
		if (keyboard[SDL_SCANCODE_LEFT])
			cameraOrientation = add(cameraOrientation, NewVector3(0, -50*rotationSpeed, 0));
		if (keyboard[SDL_SCANCODE_RIGHT])
			cameraOrientation = add(cameraOrientation, NewVector3(0, 50*rotationSpeed, 0));
		if (mouseRotation)
			cameraOrientation = add(cameraOrientation, NewVector3(rotationSpeed * mouseDeltaY, rotationSpeed * mouseDeltaX, 0));
		cameraForward = rotate(forward, cameraOrientation);
		cameraRight = rotate(right, cameraOrientation);
		cameraUp = rotate(up, cameraOrientation);
		if (keyboard[SDL_SCANCODE_W])
			cameraPos = add(cameraPos, mul_s(movementSpeed, cameraForward));
		if (keyboard[SDL_SCANCODE_S])
			cameraPos = add(cameraPos, mul_s(-movementSpeed, cameraForward));
		if (keyboard[SDL_SCANCODE_A])
			cameraPos = add(cameraPos, mul_s(-movementSpeed, cameraRight));
		if (keyboard[SDL_SCANCODE_D])
			cameraPos = add(cameraPos, mul_s(movementSpeed, cameraRight));
		if (keyboard[SDL_SCANCODE_E])
			cameraPos = add(cameraPos, mul_s(movementSpeed, cameraUp));
		if (keyboard[SDL_SCANCODE_Q])
			cameraPos = add(cameraPos, mul_s(-movementSpeed, cameraUp));
		cameraPos = clamp(cameraPos, NewVector3(0, 0, 0), NewVector3(1, 1, 1));
		if (keyboard[SDL_SCANCODE_F])
			focalLength += 5;
		if (keyboard[SDL_SCANCODE_G])
			focalLength -= 5;
		UpdateCameraParameters(cameraPos, cameraOrientation, focalLength);

		// Render and display scene
		Render(type);
		SDL_UpdateTexture(sdlTexture, NULL, camera->pixels, camera->resolution.y * sizeof(uint32_t));
		SDL_RenderClear(sdlRenderer);
		SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, NULL);
		SDL_RenderPresent(sdlRenderer);
		
		// Delta time
		t = clock() - t;
		delta_time = ((double)t) / CLOCKS_PER_SEC; // in seconds
		printf("Camera Position: %s\n", Vector3ToString(cameraPos));
		printf("Camera Orientation: %s\n", Vector3ToString(cameraOrientation));
		printf("Delta time: %.2lf seconds (%.2lf fps)\n\n", delta_time, 1 / delta_time);
		sprintf(title, "%s %d fps\n", RenderTypeToString[type], (int)(1 / delta_time));
		SDL_SetWindowTitle(sdlWindow, title);
	}
#endif

	printf("Rendering\n");
	t = clock();
	Render(type);

#ifdef USE_SDL
	sdlFlag = fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : SDL_WINDOW_RESIZABLE;
	SDL_CreateWindowAndRenderer(700, 700, sdlFlag, &sdlWindow, &sdlRenderer);
	SDL_SetWindowTitle(sdlWindow, RenderTypeToString[type]);
	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear"); // make the scaled rendering look smoother
	SDL_RenderSetLogicalSize(sdlRenderer, camera->resolution.x, camera->resolution.y);
	sdlTexture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, camera->resolution.x, camera->resolution.y);
	SDL_UpdateTexture(sdlTexture, NULL, camera->pixels, camera->resolution.y * sizeof(uint32_t));
	SDL_RenderClear(sdlRenderer);
	SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, NULL);
	SDL_RenderPresent(sdlRenderer);
#endif

	t = clock() - t;
	delta_time = ((double)t) / CLOCKS_PER_SEC;
	printf("Total computation time: %.2lf seconds (%.2lf minutes)\n", delta_time, delta_time / 60);
	SaveImageToFile(camera->pixels, camera->resolution, "image.ppm");

#ifdef USE_SDL
	SDL_WaitKey();
	CloseWindowAndQuit(sdlWindow);
#endif
	return 0;
}
