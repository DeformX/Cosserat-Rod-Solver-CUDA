
// Jerry Hsu, jerry.hsu.research@gmail.com, 2025

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#ifdef YARNBALL_HEADLESS
// Local patch (ropesim vendoring): headless build. The KittenEngine.h umbrella
// header drags in glad/GLFW/imgui/assimp/freetype at compile time, but the
// solver only touches GL behind `if (glGetString)` runtime guards (never taken
// without a GL context). Include the solver-facing modules directly and
// forward-declare the GL-side classes, which are only ever held as pointers.
#include "KittenEngine/includes/modules/Common.h"
#include "KittenEngine/includes/modules/Rotor.h"
#include "KittenEngine/KittenGpuLBVH/lbvh.cuh"
#include "KittenEngine/includes/modules/Bound.h"
#include "KittenEngine/includes/modules/SymMat.h"
namespace Kitten {
	class Mesh;
	class ComputeBuffer;
}
namespace Kit = Kitten;
#else
#include "KittenEngine/includes/KittenEngine.h"
#include "KittenEngine/KittenGpuLBVH/lbvh.cuh"
#include "KittenEngine/includes/modules/Bound.h"
#include "KittenEngine/includes/modules/SymMat.h"
#endif

namespace YarnBall {
	using namespace glm;

	// This should really NEVER be exceeded.
	constexpr int MAX_COLLISIONS_PER_SEGMENT = 128;

	// BCC file header
	struct BCCHeader {
		char sign[3];
		unsigned char byteCount;
		char curveType[2];
		char dimensions;
		char upDimension;
		uint64_t curveCount;
		uint64_t totalControlPointCount;
		char fileInfo[40];
	};

	enum class VertexFlags {
		hasPrev = 1,			// Whether the vertex has a previous vertex
		hasNext = 2,			// Whether the vertex has a next vertex
		hasNextOrientation = 4,	// Whether the segment has a next segment
		fixOrientation = 8,		// Fix the orientation of the segment
		colliding = 16,			// Whether this is colliding (unused)
	};

	inline bool hasFlag(const uint32_t flags, const VertexFlags flag) {
		return (flags & (uint32_t)flag) != 0;
	}

	inline uint32_t setFlag(const uint32_t flags, const VertexFlags flag, const bool state) {
		return state ? flags | (uint32_t)flag : flags & ~(uint32_t)flag;
	}

	// Simulation vertex (aligned to openGL layout)
	// This includes everything needed to form the local hessian minus collisions
	typedef struct {
		// Linear 
		vec3 pos;			// Node position
		float invMass;		// Inverse nodal mass

		float lRest;		// Rest length
		float kStretch;		// Stretching stiffness
		int connectionIndex;// Index of connected node -1 if none. (Used to connect vertices)
		uint32_t flags;		// Flags see VertexFlags
	} Vertex;

	typedef struct {
		Vertex* d_verts;		// Device vertex array pointer
		Kit::Rotor* d_qs;
		vec4* d_qRests;

		vec3* d_dx;				// Temporary delta position iterants. Stored as deltas for precision.
		vec3* d_dxNext;				// Separate position output for deterministic sweeps.
		Kit::Rotor* d_qsPrev;		// Orientation snapshot for deterministic sweeps.
		vec3* d_vels;			// Velocity
		vec3* d_lastVels;		// Velocity from the last frame

		vec3* d_lastPos;		// Last vertex positions. Temp storage to speed up memory access.
		uint32_t* d_lastFlags;	// Last vertex flags. Temp storage to speed up memory access.
		int* d_lastCID;			// Last cid. Temp storage to speed up memory access.

		int* d_numCols;					// Number of collisions for each segment
		float* d_maxStepSize;			// Max step size for the segment
		float* d_paddingSize;			// Max step size for the segment
		int* d_collisions;				// Collisions IDs stored as the other segment index.
		Kit::LBVH::aabb* d_bounds;		// AABBs
		ivec2* d_boundColList;			// Colliding segment AABB IDs. 

		vec3 gravity;			// Gravity
		int numItr;				// Number of iterations used per time step

		float h;				// Time step (automatically set)
		float lastH;			// Last time step
		float time;				// Current time
		int numVerts;			// Number of vertices

		float damping;			// Damping forces
		float drag;				// Velocity decay
		// Local patch (ropesim): quadratic cross-flow air drag (Morison form).
		// dragQuad = .5*rho_air*Cd_n*d/lambda [1/m] acts on the segment-normal
		// velocity component; dragQuadT is the tangential (skin) coefficient
		// on the segment-tangential component (0 = off; = dragQuad gives a
		// component-isotropic quadratic model; inert while dragQuad == 0).
		// dragQuad == 0 (default) takes the legacy path with an unchanged
		// arithmetic DAG (bitwise-identical output expected).
		// See sim/iteration.cu endItr.
		float dragQuad;			// Quadratic normal drag coefficient [1/m]
		float dragQuadT;		// Quadratic tangential drag coefficient [1/m]
		// Local patch (ropesim): rod-plane contact. Half-spaces n.x >= w with
		// planes[i] = (n, w), n unit. Vertex-local penalty with stiffness
		// k_i = m_i * (kappa / h)^2 (mass-consistent; bounce independent of h)
		// + regularized Coulomb friction with PyElastica's static->kinetic slip
		// blend, implicit in cosseratItr. numPlanes == 0 (default) skips it.
		int numPlanes;
		vec4 planes[8];
		float contactRadius;	// rope radius R [m]
		float contactKappa;	// contact stiffness: k_i = m_i * (kappa / h)^2
		float contactMuS;	// static friction coefficient
		float contactMuK;	// kinetic friction coefficient
		float contactVSlip;	// slip speed threshold [m/s] (> 0)
		// Local patch (ropesim): rod-mesh contact through signed-distance grids.
		// sdf[m].v holds phi on an (nx, ny, nz) node lattice (C order, i slowest)
		// at origin + (i, j, k) / invH in the object's own frame; the object's pose
		// per env is d_sdfPoses[env * kMaxSdf + m] (x_world = q * p + pos), env =
		// vertex / sdfEnvVerts. Contact when phi(x) < contactRadius, with the
		// plane term (normal = grad phi). numSdf == 0 (default) skips it.
		struct SdfGrid { const float* v; int nx, ny, nz; vec3 origin; float h, invH; float muS, muK; };	// + its own friction
		// Pose of (env, mesh) at the END of the current substep plus the object's
		// velocity over it (vel of the origin, omega: world frame). The material
		// point under a vertex moves h * (vel + omega x (x - pos)) this substep;
		// friction acts on the vertex displacement relative to that (sdfMoving).
		// lo/hi: world AABB of the lattice at that pose (broad phase).
		struct SdfPose { vec3 pos; vec4 q; vec3 vel; vec3 omega; vec3 lo, hi; };
		// One frame of kinematic motion: from (p0, q0) at the frame start to
		// p0 + dp, exp(rot) * q0 at its end (rot = rotation vector of q1 q0^-1 in
		// the world frame, shortest arc). While sdfMoving != 0, initItr writes the
		// SdfPose of substep s of S (pose at (s + 1) / S, constant velocity).
		struct SdfMotion { vec3 p0; vec4 q0; vec3 dp; vec3 rot; };
		enum { kMaxSdf = 64, kMaxSdfCand = 8 };	// MetaData is an unnamed struct: no static members
		int numSdf;
		SdfGrid sdf[kMaxSdf];
		SdfPose* d_sdfPoses;			// [sdfEnvs * kMaxSdf]
		const SdfMotion* d_sdfMotion;	// [sdfEnvs * kMaxSdf], read while sdfMoving
		int sdfMoving;
		int sdfEnvs;
		int sdfEnvVerts;
		// Broad phase: endItr lists, per vertex, the meshes of its env whose
		// world AABB (padded by the vertex's next-substep travel) contains it;
		// the Newton iterations and the readout visit only those. -1 = overflow
		// (or a fresh frame): visit every mesh of the env.
		int* d_sdfCand;					// [numVerts * kMaxSdfCand]
		int* d_sdfNumCand;				// [numVerts]
		// Contact-force readout (sdfForces != 0): endItr accumulates, per substep,
		// the force every surface applied to each vertex (d_contactForce
		// [numVerts], world) and the wrench the rope applied to each (env, mesh)
		// about the mesh origin (d_sdfWrench [sdfEnvs * kMaxSdf * 2]: force,
		// torque). The host zeroes them at the frame start and averages.
		int sdfForces;
		vec3* d_contactForce;
		vec3* d_sdfWrench;
		float frictionCoeff;	// Friction coefficient for contacts
		float kCollision;		// Stiffness of the collision
		float kFriction;		// Stiffness of the collision

		float detectionRadius;			// Total detection radius of the yarn (automatically set)
		float scaledDetectionRadius;	// Detection radius scaled by the detectionScaler
		float radius;					// Yarn radius. Note that this is the minimum radius. The actual radius is r + 0.5 * barrierThickness
		float accelerationRatio;		// Solver acceleration ratio

		float barrierThickness;	// Collision energy barrier thickness. This is the barrier between yarns.
		float detectionScaler;	// The extra room needed for a close by potential collision to be added as a ratio

		float bvhRebuildPeriod;	// The time in between rebuilding the BVH.
		int detectionPeriod;	// The number of steps in between to perform collision detection. -1 to turn off collisions

		float maxSegLen;			// Largest segment length
		float minSegLen;			// Largest segment length
		int useStepSizeLimit;		// Whether to use the step size limit
	} MetaData;

	class Sim {
		// Local addition (ropesim integration): lets the ropesim batch driver
		// run the substep loop itself (rebuildCUDAGraph/uploadMeta/
		// startIterate + graph launch) so it can inject a root-motion-program
		// kernel between substeps (quadratic position sweep + SLERPed
		// orientation). Zero behavior change for all other users.
		friend class SimAccess;

	public:
		enum {
			ERROR_NONE = 0,
			ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED,
			WARNING_SEGMENT_INTERPENETRATION
		};

		Vertex* verts;
		Kit::Rotor* qs;		// Quaternions orientations
		vec4* qRests;		// Rest angles
		vec3* vels;			// Velocity

		MetaData meta;
		float maxH = 1e-3;						// Largest time step allowed
		std::vector<float> initialInvMasses;	// Starting inverse masses saved for pinned vertices.
		Kit::Bound<> currentBounds;				// Current bounding box

		int lastErrorCode = ERROR_NONE;
		int lastWarningCode = ERROR_NONE;

		bool printErrors = true;
		bool renderShaded = false;

	private:
		const bool deterministicSweeps;
		MetaData* d_meta = nullptr;
		int* d_error = nullptr;
		bool initialized = false;
		Kit::LBVH bvh;

		float lastBVHRebuild = std::numeric_limits<float>::infinity();
		int lastItr = -1;
		size_t stepCounter = 0;

		// GL stuff
		Kitten::Mesh* cylMesh = nullptr;
		Kitten::Mesh* cylMeshHiRes = nullptr;
		Kitten::ComputeBuffer* vertBuffer = nullptr;
		Kitten::ComputeBuffer* qBuffer = nullptr;

		cudaStream_t stream = nullptr;
		cudaGraphExec_t stepGraph = nullptr;

	public:
		Sim(int numVerts, bool deterministicSweeps = false);
		~Sim();

		inline size_t stepCount() { return stepCounter; }

		// Local addition (ropesim integration): expose the solver stream so
		// external kernels (batched root driving, observation gather) can be
		// enqueued on it — ordered with the solve, no extra host syncs.
		inline cudaStream_t solverStream() const { return stream; }

		// Initializes memory and sets up rest length, angles, and mass
		void configure(float density = 1e-3);
		void setKBend(float k = 3e-9);
		void setKStretch(float k = 1e-2);

		// Rendering
		void startRender();
		void render();
		void renderShadows();

		// Utils
		// void glueEndpoints(float searchRadius);
		void upload();			// Upload the CPU data to the GPU
		void download();		// Download the GPU data to the CPU
		void zeroVelocities();

		// Simulation
		void step(float dt);		// Perform one timestep
		float advance(float dt);	// Advance the simulation by dt using one or more timesteps.

		void printCollisionStats();
		Kitten::LBVH::aabb bounds();

		void exportFiberMesh(std::string path);
		void exportToBCC(std::string path, bool exportAsPolyline = false);
		void exportToOBJ(std::string path);

		// Glue endpoints with a vertex within the search radius
		void glueEndpoints(float searchRadius);

	private:
		void uploadMeta();

		// Local patch (ropesim): substep index / count of the current frame
		// (kinematic mesh poses are interpolated on them).
		void startIterate(int sub = 0, int nSub = 1);
		void endIterate();
		void detectCollisions();
		void iterateCosserat();
		void recomputeStepLimit();
		void checkErrors();

		void rebuildCUDAGraph();
	};

	Sim* readFromOBJ(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves = false, bool allowResample = true);
	Sim* readFromBCC(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves = false, bool allowResample = true);
	Sim* readFromPoly(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves = false, bool allowResample = true);
	Sim* buildFromJSON(std::string path);
}