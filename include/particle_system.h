#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "data_structures.h"


/** @brief Frees memory allocated for a ParticleSystem on the host.
 *
 * This function frees all memory associated with the ParticleSystem structure.
 * @param ps Reference to the ParticleSystem structure to be freed.
 */
void ps_freeMem_device(ParticleSystem* d_ps);

ParticleSystem* ps_allocate_device(int num_particles);

#endif // PARTICLE_SYSTEM_H