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
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static const double EPS = 1e-3;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  //create normal distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 50;
  //initialize all weights to 1.0
  weights.resize(num_particles, 1.0);
  particles.resize(num_particles);

  for (int i = 0; i < particles.size(); ++i)
  {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  for (auto& p : particles)
  {
    auto yawd = yaw_rate * delta_t;
    if (fabs(yaw_rate) < EPS)
    {
        p.x += velocity*delta_t*cos(p.theta);
        p.y += velocity*delta_t*sin(p.theta);
    } else { //when yaw_Rate is not zero
        p.x += velocity/yaw_rate * (sin(p.theta + yawd) - sin(p.theta));
        p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yawd));
        p.theta += yawd;
    }

    //sample from normal distribution
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);
    //add gaussian noise to position
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  //for each observations find the closest predicted landmark
  for (auto& observation : observations)
  {
    double min_distance = INFINITY;
    int index = 0;
    for (int i = 0; i < predicted.size(); ++i)
    {
      double dx = (predicted[i].x - observation.x);
      double dy = (predicted[i].y - observation.y);
      double dist = dx*dx + dy*dy;
      if (dist < min_distance)
      {
        min_distance = dist;
        index = i;
      }
    }
    observation.id = index;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  for (int i = 0; i < particles.size(); ++i)
  {
    Particle& p = particles[i];
    //convert observations from local to world coordinates
    vector<LandmarkObs> worldObservations(observations);
    for (auto& obs : worldObservations)
    {
        double wx = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
        double wy = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
        obs.x = wx;
        obs.y = wy;
    }
    //find list of landmarks that are within the range to the particle
    vector<LandmarkObs> predicted;
    for (auto& lm : map_landmarks.landmark_list)
    {
      double dx = lm.x_f - p.x;
      double dy = lm.y_f - p.y;

      if (dx*dx + dy*dy <= sensor_range * sensor_range)
      {
          LandmarkObs lmp;
          lmp.x = lm.x_f;
          lmp.y = lm.y_f;
          lmp.id = lm.id_i;
          predicted.push_back(lmp);
      }
    }
    //find the closest landmark and associated w the corresponding observations
    dataAssociation(predicted, worldObservations);

    //calculates the probability for this particle given the observations
    double prob = 1.0;
    for (auto& obs : worldObservations)
    {
      auto& lm = predicted[obs.id];
      //calculates prob assuming a 2d gaussian distribution
      double varx = std_landmark[0] * std_landmark[0];
      double vary = std_landmark[1] * std_landmark[1];
      double denom = 2.0*M_PI*std_landmark[0]*std_landmark[1]; 
      double dx = obs.x - lm.x;
      double dy = obs.y - lm.y;
      prob *= exp(-(dx*dx/(2.0*varx) + dy*dy/(2.0*vary)))/denom; 
    }
    p.weight = prob;
    weights[i] = prob; 
  }
}

void ParticleFilter::resample() {
  //produce random integer in the interval [0, n] where n is number of
  //particles according to its weights
  discrete_distribution<int> w(weights.begin(), weights.end());
  vector<Particle> resampledParticles;

  for (int i = 0; i < particles.size(); ++i)
  {
     resampledParticles.push_back(particles[w(gen)]);
  }
  particles = std::move(resampledParticles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
