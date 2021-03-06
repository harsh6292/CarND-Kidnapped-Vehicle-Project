/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Let num of particles be 100
    num_particles = 100;

    std::default_random_engine random_gen;

    // Create a guassian noise distribution of particle position around GPS position
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Create each particle with a random distribution around GPS coordinates
    for (int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(random_gen);
        particle.y = dist_y(random_gen);
        particle.theta = dist_theta(random_gen);
        particle.weight = 1;

        particles.push_back(particle);
        weights.push_back(1);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine random_gen;

    // Update particles position and heading
    for (int i = 0; i < num_particles; i++) {
        double pred_x, pred_y, pred_theta;

        // Update current particle position based on velocity and yaw rate observed over time delta_t
        if (yaw_rate == 0) {
            pred_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            pred_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
            pred_theta = particles[i].theta;
        } else {
            pred_x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            pred_y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            pred_theta = particles[i].theta + yaw_rate*delta_t;
        }


        // Add noise to prediction
        normal_distribution<double> dist_x(pred_x, std_pos[0]);
        normal_distribution<double> dist_y(pred_y, std_pos[1]);
        normal_distribution<double> dist_theta(pred_theta, std_pos[2]);

        particles[i].x = dist_x(random_gen);
        particles[i].y = dist_y(random_gen);
        particles[i].theta = dist_theta(random_gen);
    }

}

int ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> map_landmark_list, LandmarkObs& single_observation) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    double dist_obs_landmark = std::numeric_limits<double>::max();
    int min_dist_landmark_index = 0;

    // For each landmark position, find the observation closest to it
    for(int i = 0; i < map_landmark_list.size(); i++) {
        double euclid_dist = dist(single_observation.x, single_observation.y, map_landmark_list[i].x_f, map_landmark_list[i].y_f);

        if (euclid_dist < dist_obs_landmark) {
            dist_obs_landmark = euclid_dist;
            min_dist_landmark_index = i;
        }
    }

    return min_dist_landmark_index;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // Clear the weights of previous observations particles
    weights.clear();

    // For each particle, transform the observations from vehicle coordinates to particle (map) coordinates
    for (int i = 0; i < num_particles; i++) {

        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;

        // Reset each particle weight to be 1 as we are updating based on current observations
        particles[i].weight = 1.0;

        // Tranform each observation
        for (LandmarkObs obs : observations) {

            // Temporary observations that will store transformed observations
            LandmarkObs temp_obs;

            // Calculate transformations and rotations
            temp_obs.id = obs.id;
            double sin_theta = sin(particles[i].theta);
            double cos_theta = cos(particles[i].theta);
            temp_obs.x = particles[i].x + (cos_theta * obs.x - sin_theta * obs.y);
            temp_obs.y = particles[i].y + (sin_theta * obs.x + cos_theta * obs.y);

            // Push the observed measurements
            sense_x.push_back(temp_obs.x);
            sense_y.push_back(temp_obs.y);


            // Find the nearest landmark to observed position
            int min_dist_landmark_index = dataAssociation(map_landmarks.landmark_list, temp_obs);

            // Extract the nearest landmark from list of map landmarks
            Map::single_landmark_s nearest_landmark = map_landmarks.landmark_list[min_dist_landmark_index];

            // Assign the ID of map landmark to nearest observations for later visualization in main
            temp_obs.id = nearest_landmark.id_i;

            // Push the associated landmark assoicated with observation
            associations.push_back(temp_obs.id);


            // For each observation, calculate weight
            double denom = 2 * M_PI * std_landmark[0] * std_landmark[1];
            double weight_each_obs = 0.0;

            double exp_power_term_1 = (pow((temp_obs.x - nearest_landmark.x_f), 2)/(2*std_landmark[0]*std_landmark[0]));
            double exp_power_term_2 = (pow((temp_obs.y - nearest_landmark.y_f), 2)/(2*std_landmark[1]*std_landmark[1]));

            weight_each_obs = (1/denom) * exp(-1.0 * (exp_power_term_1 + exp_power_term_2));


            // Multiply each weight of observation to determine probability the particle saw the correct landmark
            if (weight_each_obs > 0) {
                particles[i].weight *= weight_each_obs;
            }
        }


        // Push the Multivariate-Guassian probability for this particle
        weights.push_back(particles[i].weight);

        // Update associations of the particle
        SetAssociations(particles[i], associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    // Initialize the distribution of how particles will be resampled with replacement based on their weights
    std::default_random_engine random_gen;
    std::discrete_distribution<> distribution_weights(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles;

    // Resample particles
    for (int i = 0; i < num_particles; i++) {
        resampled_particles.push_back(particles[distribution_weights(random_gen)]);
    }

    particles.clear();
    particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
