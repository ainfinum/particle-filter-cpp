/**
 * particle_filter.cpp
 * 2D particle filter
 * Andrei Sasinovich
 */
#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>

#include "helper_functions.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{

  default_random_engine gen;
  num_particles = 25;

  //Gaussian distribution
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++)
  {

    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{

  default_random_engine gen;
  double v_yaw = velocity / yaw_rate;
  double yaw_dt = yaw_rate * delta_t;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++)
  {

    if (fabs(yaw_rate) < 0.00001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {

      particles[i].x += v_yaw * (sin(particles[i].theta + yaw_dt) - sin(particles[i].theta));
      particles[i].y += v_yaw * (cos(particles[i].theta) - cos(particles[i].theta + yaw_dt));
      particles[i].theta += yaw_dt;
    }

    //Calculate Prediction
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);

    if (log)
    {
      std::cout << "particles[" << i << " ] " << particles[i].x << " " << particles[i].y << " " << particles[i].theta << std::endl;
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * Findind the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
  */

  // nearest neighbor data association
  int i, j;
  for (i = 0; i < (int)observations.size(); i++)
  {
    double min_dist = numeric_limits<double>::max();
    int closest_id = -1;

    for (j = 0; j < (int)predicted.size(); j++)
    {
      double current_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (current_dist < min_dist)
      {
        min_dist = current_dist;
        closest_id = predicted[j].id;
      }
    }

    observations[i].id = closest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * Updating the weights of each particle using a mult-variate Gaussian 
   *   distribution. 
   */

  int i, j;

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double total_weight = 0.0;

  weights.clear();

  for (i = 0; i < num_particles; i++)
  {

    // Transform to MAP'S coordinate system
    vector<LandmarkObs> observations_maps_cs;
    for (j = 0; j < (int)observations.size(); j++)
    {
      double x_map = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
      double y_map = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
      observations_maps_cs.push_back(LandmarkObs{observations[j].id, x_map, y_map});
    }

    //Search landmarks in sensor range
    vector<LandmarkObs> landmarks_in_range;
    for (j = 0; j < (int)map_landmarks.landmark_list.size(); j++)
    {
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      double x_dist = landmark_x - particles[i].x;
      double y_dist = landmark_y - particles[i].y;

      if (fabs(x_dist) <= sensor_range && fabs(y_dist) <= sensor_range)
      {
        landmarks_in_range.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, landmark_x, landmark_y});
        if (log)
        {
          std::cout << "landmarks_in_range[" << map_landmarks.landmark_list[j].id_i << "] " << landmark_x << "," << landmark_y << std::endl;
        }
      }
    }

    if (log)
    {
      std::cout << "landmarks_in_range.size: " << landmarks_in_range.size() << std::endl;
    }

    dataAssociation(landmarks_in_range, observations_maps_cs);

    // calculate normalization term
    double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
    double pow_sig_x = 2 * pow(sig_x, 2);
    double pow_sig_y = 2 * pow(sig_y, 2);
    double weight = 1.0;
    particles[i].weight = 1.0;

    for (int m = 0; m < (int)observations_maps_cs.size(); m++)
    {
      double mu_x = 0.0;
      double mu_y = 0.0;

      for (int n = 0; n < (int)landmarks_in_range.size(); n++)
      {
        if (landmarks_in_range[n].id == observations_maps_cs[m].id)
        {
          mu_x = landmarks_in_range[n].x;
          mu_y = landmarks_in_range[n].y;

          // calculate exponent
          double exponent = (pow(observations_maps_cs[m].x - mu_x, 2) / pow_sig_x) + (pow(observations_maps_cs[m].y - mu_y, 2) / pow_sig_y);
          weight = gauss_norm * exp(-exponent);

          if (weight <= 0.00000001)
          {
            weight = 0.00000001;
          }

          particles[i].weight *= weight;
        }
      }

      if (log)
      {
        std::cout << "m: " << m << " weight " << weight << std::endl;
        std::cout << "particles[" << i << "].weight: " << particles[i].weight << std::endl;
      }
    }

    //Update total weight for normalization
    total_weight += particles[i].weight;
    if (log)
    {
      std::cout << "particles.weight[" << i << "]= " << particles[i].weight << " total_weight= " << total_weight << std::endl;
    }
  } //end particles

  //Normalize the weights of all particles
  for (i = 0; i < num_particles; i++)
  {
    particles[i].weight /= total_weight;
    weights.push_back(particles[i].weight);
  }
}

void ParticleFilter::resample()
{
  /**
   *  Resampling particles with replacement with probability proportional 
   *   to their weight. 
   */

  double max_w = *max_element(weights.begin(), weights.end());

  default_random_engine gen;
  uniform_int_distribution<int> dist_index(0, num_particles - 1);
  int index = dist_index(gen);
  double beta = 0.0;
  uniform_real_distribution<double> dist_beta(0.0, max_w);
  vector<Particle> p3;

  for (int i = 0; i < num_particles; i++)
  {
    beta += dist_beta(gen) * 2.0;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    p3.push_back(particles[index]);
  }

  particles = p3;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}