# LPR-Survey is updating
A survey of LiDAR-based place recognition 

# Global Descriptors
## Handcrafted Methods
### BEV-based Methods
#### SC Family

 1. Scan context: Egocentric spatial descriptor for place recognition within 3d point cloud map, **IROS 2018**.
 2. Scan Context++: Structural Place Recognition Robust to Rotation and Lateral Variations in Urban Environments, **TRO 2021**.
 3. Intensity scan context: Coding intensity and geometry relations for loop closure detection, **ICRA 2020**.
 4. Weighted scan context: Global descriptor with sparse height feature for loop closure detection, **ICCCR 2021**.
 5. Fresco: Frequency-domain scan context for lidar-based place recognition with translation and rotation invariance, **ICARCV 2022**.
 6. Global place recognition using an improved scan context for lidar-based localization system, **AIM 2021**.
 7. High accuracy and low complexity lidar place recognition using unitary invariant frobenius norm, **IEEE Sensors Journal 2022**.
 8. Place recognition of large-scale unstructured orchards with attention score maps, **RAL 2023**.
 9. Art-slam: Accurate real-time 6dof lidar slam, **RAL 2022**.
 10. Optimized sc-f-loam: Optimized fast lidar odometry and mapping using scan context, CVCI 2022.
 11. Eil-slam: Depth-enhanced edge-based infrared-lidar slam, **JFR 2022**.
 
 #### Pairwise Matching
1. Lidar iris for loop-closure detection, **IROS 2020**.
 2. A heterogeneous 3d map-based place recognition solution using virtual lidar and a polar grid height coding image descriptor, **ISPRS JPRS 2022**.
 3. One ring to rule them all: Radon sinogram for place recognition, orientation and translation estimation, **IROS 2022**.
 4. Ring++: Roto-translation invariant gram for global localization on a sparse scan map, **arXiv 2022**. 

### Discretization-based Methods
#### Fixed-size Discretization
1. M. Magnusson, H. Andreasson, A. Nuchter, and A. J. Lilienthal, “Appearance-based loop detection from 3d laser data using the normal distributions transform,” in 2009 IEEE International Conference on Robotics and Automation. IEEE, 2009, pp. 23–28.
2. M. Magnusson, T. P. Kucner, S. G. Shahbandi, H. Andreasson, and A. J. Lilienthal, “Semi-supervised 3d place categorisation by descriptor clustering,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017, pp. 620–625.
3. J. Lin and F. Zhang, “A fast, complete, point cloud based loop closure for lidar odometry and mapping,” arXiv preprint arXiv:1909.11811, 2019
4. Q. Meng, H. Guo, X. Zhao, D. Cao, and H. Chen, “Loopclosure detection with a multiresolution point cloud histogram mode in lidar odometry and mapping for intelligent vehicles,” IEEE/ASME Transactions on Mechatronics, vol. 26, no. 3, pp. 1307– 1317, 2021.
5. F. Cao, F. Yan, S. Wang, Y. Zhuang, and W. Wang, “Season-invariant and viewpoint-tolerant lidar place recognition in gps-denied environments,” IEEE Transactions on Industrial Electronics, vol. 68, no. 1, pp. 563–574, 2021.
#### Unfixed-size Discretization
1. K. P. Cop, P. V. Borges, and R. Dube, “Delight: An efficient ´ descriptor for global localisation using lidar intensities,” in 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 3653–3660.
2. J. Mo and J. Sattar, “A fast and robust place recognition approach for stereo visual odometry using lidar descriptors,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020, pp. 5893–5900.
### Point-based Methods
1. N. Muhammad and S. Lacroix, “Loop closure detection using small-sized signatures from 3d lidar data,” in 2011 IEEE International Symposium on Safety, Security, and Rescue Robotics. IEEE, 2011, pp. 333–338.
2. T. Rohling, J. Mack, and D. Schulz, “A fast histogram-based ¨ similarity measure for detecting loop closures in 3-d lidar data,” in 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2015, pp. 736–741.
3. L. He, X. Wang, and H. Zhang, “M2dp: A novel 3d point cloud descriptor and its application in loop closure detection,” in 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016, pp. 231–237.
4. L. Perdomo, D. Pittol, M. Mantelli, R. Maffei, M. Kolberg, and E. Prestes, “c-m2dp: A fast point cloud descriptor with color information to perform loop closure detection,” in 2019 IEEE 15th International Conference on Automation Science and Engineering (CASE). IEEE, 2019, pp. 1145–1150
## Learning-based Methods
