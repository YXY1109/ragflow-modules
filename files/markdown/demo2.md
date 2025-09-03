# Real-time Temporal Stereo Matchingusing Iterative Adaptive Support Weights

J˛edrzej Kowalczuk, Eric T. Psota, and Lance C. PérezDepartment of Electrical Engineering, University of Nebraska-Lincoln[jkowalczuk2,epsota,lperez]@unl.edu

Abstract—Stereo matching algorithms are nearly always de-signed to find matches between a single pair of images. A methodis presented that was specifically designed to operate on sequencesof images. This method considers the cost of matching imagepoints in both the spatial and temporal domain. To maintainreal-time operation, a temporal cost aggregation method is usedto evaluate the likelihood of matches that is invariant with respectto the number of prior images being considered. This methodhas been implemented on massively parallel GPU hardware,and the implementation ranks as one of the fastest and mostaccurate real-time stereo matching methods as measured by theMiddlebury stereo performance benchmark.

# I. INTRODUCTION

Modern stereo matching algorithms achieve excellent resultson static stereo images, as demonstrated by the Middleburystereo performance benchmark [1], [2]. However, their ap-plication to stereo video sequences does not guarantee inter-frame consistency of matches extracted from subsequent stereoframe pairs. The lack of temporal consistency of matchesbetween successive frames introduces spurious artifacts in theresulting disparity maps. The problem of obtaining temporallyconsistent sequences of disparity maps from video streams isknown as the temporal stereo correspondence problem, yetthe amount of research efforts oriented towards finding aneffective solution to this problem is surprisingly small.

A method is proposed for real-time temporal stereo match-ing that efficiently propagates matching cost information be-tween consecutive frames of a stereo video sequence. Thismethod is invariant to the number of prior frames beingconsidered, and can be easily incorporated into any local stereomethod based on edge-aware filters. The iterative adaptivesupport matching algorithm presented in [3] serves as afoundation for the proposed method.

# II. BACKGROUND

Stereo matching is the process of identifying correspon-dences between pixels in stereo images obtained using apair of synchronized cameras. These correspondences areconveniently represented using the notion of disparity, i.e. thepositional offset between two matching pixels. It is assumedthat the stereo images are rectified, such that matching pixelsare confined within corresponding rows of the images andthus disparities are restricted to the horizontal dimension, asillustrated in Figure 1. For visualization purposes, disparitiesrecovered for every pixel of a reference image are storedtogether in the form of an image, which is known as thedisparity map. Note that individual disparities can be convertedto actual depths if the geometry of the camera setup isknown, i.e., the stereo configuration of cameras has been pre-calibrated.

![](http://127.0.0.1:9000/mineru/images/60ec91bbbd047fbafd8b25eb3606bbea75cbb6cec0e3a59f67611fd3403fb52e.jpg)  
Figure 1: Geometry of two horizontally aligned views where $\mathbf { p }$denotes a pixel in the reference frame, $\bar { \bf p }$ denotes its matchingpixel in the target frame, and $d _ { p }$ denotes the disparity betweenthem along the horizontal dimension.

In their excellent taxonomy paper [1], Scharstein andSzeliski classify stereo algorithms as local or global meth-ods. Global methods, which offer outstanding accuracy, aretypically derived from an energy minimization frameworkthat allows for explicit integration of disparity smoothnessconstraints and thus is capable of regularizing the solutionin weakly textured areas. The minimization, however, is oftenachieved using iterative methods or graph cuts, which do notlend themselves well to parallel implementation.

In contrast, local methods, which are typically built uponthe Winner-Takes-All (WTA) framework, have the property ofcomputational regularity and are thus suitable for implemen-tation on parallel graphics hardware. Within the WTA frame-work, local stereo algorithms consider a range of disparityhypotheses and compute a volume of pixel-wise dissimilaritymetrics between the reference image and the matched image atevery considered disparity value. Final disparities are chosenfrom the cost volume by traversing through its values andselecting the disparities associated with minimum matchingcosts for every pixel of the reference image.

Disparity maps obtained using this simple strategy are oftentoo noisy to be considered useable. To reduce the effectsof noise and enforce spatial consistency of matches, localstereo algorithms consider arbitrarily shaped and sized supportwindows centered at each pixel of the reference image, andaggregate cost values within the pixel neighborhoods definedby these windows. In 2005, Yoon and Kweon [4] proposedan adaptive matching cost aggregation scheme, which assignsa weight value to every pixel located in the support windowof a given pixel of interest. The weight value is based onthe spatial and color similarity between the pixel of interestand a pixel in its support window, and the aggregated cost iscomputed as a weighted average of the pixel-wise costs withinthe considered support window. The edge-preserving natureand matching accuracy of adaptive support weights have madethem one of the most popular choices for cost aggregation inrecently proposed stereo matching algorithms [3], [5]–[8].

Recently, Rheman et al. [9], [10] have revisited the costaggregation step of stereo algorithms, and demonstrated thatcost aggregation can be performed by filtering of subsequentlayers of the initially computed matching cost volume. In par-ticular, the edge-aware image filters, such as the bilateral filterof Tomasi and Manducci [11] or the guided filter of He [12],have been rendered useful for the problem of matching costaggregation, enabling stereo algorithms to correctly recoverdisparities along object boundaries. In fact, Yoon and Kweon’sadaptive support-weight cost aggregation scheme is equivalentto the application of the so-called joint bilateral filter to thelayers of the matching cost volume.

It has been demonstrated that the performance of stereoalgorithms designed to match a single pair of images canbe adapted to take advantage of the temporal dependenciesavailable in stereo video sequences. Early proposed solutionsto temporal stereo matching attempted to average matchingcosts across subsequent frames of a video sequence [13],[14]. Attempts have been made to integrate estimation ofmotion fields (optical flow) into temporal stereo matching. Themethods of [15] and [16] perform smoothing of disparitiesalong motion vectors recovered from the video sequence. Theestimation of the motion field, however, prevents real-timeimplementation, since state-of-the-art optical flow algorithmsdo not, in general, approach real-time frame rates. In a relatedapproach, Sizintsev and Wildes [17], [18] used steerablefilters to obtain descriptors characterizing motion of imagefeatures in both space and time. Unlike traditional algorithms,their method performs matching on spatio-temporal motiondescriptors, rather than on pure pixel intensity values, whichleads to improved temporal coherence of disparity maps at thecost of reduced accuracy at depth discontinuities.

Most recently, local stereo algorithms based on edge-awarefilters were extended to incorporate temporal evidence intothe matching process. The method of Richardt et al. [19]employs a variant of the bilateral grid [20] implemented ongraphics hardware, which accelerates cost aggregation andallows for weighted propagation of pixel dissimilarity metricsfrom previous frames to the current one. Although this methodoutperforms the baseline frame-to-frame approach, the amountof hardware memory necessary to construct the bilateral gridlimits its application to single-channel, i.e., grayscale imagesonly. Hosni et al. [10], on the other hand, reformulated kernelsof the guided image filter to operate on both spatial andtemporal information, making it possible to process a temporalcollection of cost volumes. The filtering operation was shownto preserve spatio-temporal edges present in the cost volumes,resulting in increased temporal consistency of disparity maps,greater robustness to image noise, and more accurate behavioraround object boundaries.

# III. METHOD

The proposed temporal stereo matching algorithm is anextension of the real-time iterative adaptive support-weightalgorithm described in [3]. In addition to real-time two-pass aggregation of the cost values in the spatial domain,the proposed algorithm enhances stereo matching on videosequences by aggregating costs along the time dimension.The operation of the algorithm has been divided into fourstages: 1) two-pass spatial cost aggregation, 2) temporal costaggregation, 3) disparity selection and confidence assessment,and 4) iterative disparity refinement. In the following, each ofthese stages is described in detail.

# A. Two-Pass Spatial Cost Aggregation

Humans group shapes by observing the geometric distanceand color similarity of points in space. To mimic this vi-sual grouping, the adaptive support-weight stereo matchingalgorithm [4] considers a support window $\Omega _ { p }$ centered at thepixel of interest $p$ , and assigns a support weight to each pixel$q \in \Omega _ { p }$ . The support weight relating pixels $p$ and $q$ is givenby

$$
w ( p , q ) = \exp { \left( - \frac { \Delta _ { g } ( p , q ) } { \gamma _ { g } } - \frac { \Delta _ { c } ( p , q ) } { \gamma _ { c } } \right) } ,
$$

where $\Delta _ { g } ( p , q )$ is the geometric distance, $\Delta _ { c } ( p , q )$ is the colordifference between pixels $p$ and $q$ , and the coefficients $\gamma _ { g }$ and$\gamma _ { c }$ regulate the strength of grouping by geometric distance andcolor similarity, respectively.

To identify a match for the pixel of interest $p$ , the real-timeiterative adaptive support-weight algorithm evaluates matchingcosts between $p$ and every match candidate $\bar { p } \in S _ { p }$ , where $S _ { p }$denotes a set of matching candidates associated with pixel $p$ .For a pair of pixels $p$ and $\bar { p }$ , and their support windows $\Omega _ { p }$and $\Omega _ { \bar { p } }$ , the initial matching cost is aggregated using

$$
C ( p , \bar { p } ) = \frac { { \displaystyle \sum _ { q \in \Omega _ { p } , \bar { q } \in \Omega _ { \bar { p } } } w ( p , q ) w ( \bar { p } , \bar { q } ) \delta ( q , \bar { q } ) } } { { \displaystyle \sum _ { q \in \Omega _ { p } , \bar { q } \in \Omega _ { \bar { p } } } w ( p , q ) w ( \bar { p } , \bar { q } ) } } ,
$$

where the pixel dissimilarity metric $\delta ( q , \bar { q } )$ is chosen as thesum of truncated absolute color differences between pixels $q$and $\bar { q }$ . Here, the truncation of color difference for the red,green, and blue components given by

$$
\delta ( q , \bar { q } ) = \sum _ { c = \{ r , g , b \} } \operatorname* { m i n } ( | q _ { c } - \bar { q } _ { c } | , \tau ) .
$$

This limits each of their magnitudes to at most $\tau$ , which pro-vides additional robustness to outliers. Rather than evaluatingEquation (2) directly, real-time algorithms often approximatethe matching cost by performing two-pass aggregation usingtwo orthogonal 1D windows [5], [6], [8]. The two-pass methodfirst aggregates matching costs in the vertical direction, andthen computes a weighted sum of the aggregated costs in thehorizontal direction. Given that support regions are of size$\omega \times \omega$ , the two-pass method reduces the complexity of costaggregation from $\mathcal { O } ( \omega ^ { 2 } )$ to $\mathcal { O } ( \omega )$ .

# B. Temporal cost aggregation

Once aggregated costs $C ( p , \bar { p } )$ have been computed for allpixels $p$ in the reference image and their respective matchingcandidates $\bar { p }$ in the target image, a single-pass temporalaggregation routine is exectuted. At each time instance, thealgorithm stores an auxiliary cost $C _ { a } ( p , \bar { p } )$ which holds aweighted summation of costs obtained in the previous frames.During temporal aggregation, the auxiliary cost is merged withthe cost obtained from the current frame using

$$
C ( p , \bar { p } ) \gets \frac { ( 1 - \lambda ) \cdot C ( p , \bar { p } ) + \lambda \cdot w _ { t } ( p , p _ { t - 1 } ) \cdot C _ { a } ( p , \bar { p } ) } { ( 1 - \lambda ) + \lambda \cdot w _ { t } ( p , p _ { t - 1 } ) } ,
$$

where the feedback coefficient $\lambda$ controls the amount of costsmoothing and $w _ { t } ( p , p _ { t - 1 } )$ enforces color similarity in thetemporal domain. The temporal adaptive weight computedbetween the pixel of interest $p$ in the current frame and pixel$p _ { t - 1 }$ , located at the same spatial coordinate in the prior frame,is given by

$$
w _ { t } ( p , p _ { t - 1 } ) = \exp { \left( - \frac { \Delta _ { c } ( p , p _ { t - 1 } ) } { \gamma _ { t } } \right) } ,
$$

where $\gamma _ { t }$ regulates the strength of grouping by color similarityin the temporal dimension. The temporal adaptive weight hasthe effect of preserving edges in the temporal domain, suchthat when a pixel coordinate transitions from one side of anedge to another in subsequent frames, the auxiliary cost isassigned a small weight and the majority of the cost is derivedfrom the current frame.

# C. Disparity Selection and Confidence Assessment

Having performed temporal cost aggregation, matches aredetermined using the Winner-Takes-All (WTA) match selec-tion criteria. The match for $p$ , denoted as $m ( p )$ , is the can-didate pixel $\bar { p } \in S _ { p }$ characterized by the minimum matchingcost, and is given by

$$
m ( p ) = \operatorname * { a r g m i n } _ { \bar { p } \in S _ { p } } C ( p , \bar { p } ) .
$$

To asses the level of confidence associated with selectingminimum cost matches, the algorithm determines another setof matches, this time from the target to reference image, andverifies if the results agree. Given that $\bar { p } = m ( p )$ , i.e. pixel $\bar { p }$in the right image is the match for pixel $p$ in the left image,and $p ^ { \prime } = m ( \bar { p } )$ , the confidence measure $F _ { p }$ is computed as

$$
F _ { p } = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m i n } _ { \bar { p } \in S _ { p } \backslash m ( p ) } C ( p , \bar { p } ) - \operatorname* { m i n } _ { \bar { p } \in S _ { p } } C ( p , \bar { p } ) } \\ { \displaystyle \operatorname* { m i n } _ { \bar { p } \in S _ { p } \backslash m ( p ) } C ( p , \bar { p } ) } ,  & { | d _ { p } - d _ { p ^ { \prime } } | \leq 1 } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. .
$$

# D. Iterative Disparity Refinement

Once the first iteration of stereo matching is complete,disparity estimates $D _ { \boldsymbol { p } } ^ { i }$ can be used to guide matching insubsequent iterations. This is done by penalizing disparitiesthat deviate from their expected values. The penalty functionis given by

$$
\Lambda ^ { i } ( p , \bar { p } ) = \alpha \times \sum _ { q \in \Omega _ { p } } w ( p , q ) F _ { q } ^ { i - 1 } \left| D _ { q } ^ { i - 1 } - d _ { p } \right| ,
$$

where the value of $\alpha$ is chosen empirically. Next, the penaltyvalues are incorporated into the matching cost as

$$
{ C } ^ { i } ( p , \bar { p } ) = { C } ^ { 0 } ( p , \bar { p } ) + { \Lambda } ^ { i } ( p , \bar { p } ) ,
$$

and the matches are reselected using the WTA match selectioncriteria. The resulting disparity maps are then post-processedusing a combination of median filtering and occlusion filling.Finally, the current cost becomes the auxiliary cost for the nextpair of frames in the video sequence, i.e., $C _ { a } ( p , \bar { p } )  C ( p , \bar { p } )$for all pixels $p$ in the and their matching candidates $\bar { p }$ .

# IV. RESULTS

The speed and accuracy of real-time stereo matching al-gorithms are traditionally demonstrated using still-frame im-ages from the Middlebury stereo benchmark [1], [2]. Stillframes, however, are insufficient for evaluating stereo match-ing algorithms that incorporate frame-to-frame prediction toenhance matching accuracy. An alternative approach is touse a stereo video sequence with a ground truth disparityfor each frame. Obtaining the ground truth disparity of realworld video sequences is a difficult undertaking due to thehigh frame rate of video and limitations in depth sensing-technology. To address the need for stereo video with groundtruth disparities, five pairs of synthetic stereo video sequencesof a computer-generated scene were given in [19]. While thesevideos incorporate a sufficient amount of movement variation,they were generated from relatively simple models using low-resolution rendering, and they do not provide occlusion ordiscontinuity maps.

To evaluate the performance of temporal aggregation, anew synthetic stereo video sequence is introduced along withcorresponding disparity maps, occlusion maps, and disconti-nuity maps for evaluating the performance of temporal stereomatching algorithms. To create the video sequence, a complexscene was constructed using Google Sketchup and a pairof animated paths were rendered photorealistically using theKerkythea rendering software. Realistic material propertieswere used to give surfaces a natural-looking appearance byadjusting their specularity, reflectance, and diffusion. Thevideo sequence has a resolution of $6 4 0 ~ \times ~ 4 8 0$ pixels, aframe rate of 30 frames per second, and a duration of 4seconds. In addition to performing photorealistic rendering,depth renders of both video sequences were also generated andconverted to ground truth disparity for the stereo video. Thevideo sequences and ground truth data have been made avail-able at http://mc2.unl.edu/current-research/image-processing/. Figure 2 shows two sample framesof the synthetic stereo scene from a single camera perspective,along with the ground truth disparity, occlusion map, anddiscontinuity map.

![](http://127.0.0.1:9000/mineru/images/abafd29f9e09d37bbaed07a45f1c3fac281749f61be8b8bea545bea8c9d0e6f9.jpg)  
Figure 2: Two sample frames from the synthetic video se-quence ( $1 ^ { \mathrm { s t } }$ row), along with their corresponding ground truthdisparity ( $2 ^ { \mathrm { n d } }$ row), occlusion map ( $3 ^ { \mathrm { r d } }$ row), and discontinuitymap $\mathrm { \cdot } 4 ^ { \mathrm { t h } }$ row).

The results of temporal stereo matching are given in Figure3 for uniform additive noise confined to the ranges of $\pm 0$ ,$\pm 2 0$ , and $\pm 4 0$ . Each performance plot is given as a functionof the feedback coefficient $\lambda$ . As with the majority of temporalstereo matching methods, improvements are negligible whenno noise is added to the images [10], [19]. This is largely dueto the fact that the video used to evaluate these methods iscomputer generated with very little noise to start with, thusthe noise suppression achieved with temporal stereo matchingshows little to no improvement over methods that operate onpairs of images.

Significant improvements in accuracy can be seen in Figure3 when the noise has ranges of $\pm 2 0$ , and $\pm 4 0$ . In this scenario,the effect of noise in the current frame is reduced by increasingthe feedback coefficient $\lambda$ . This increasing of $\lambda$ has the effectof averaging out noise in the per-pixel costs by selectingmatches based more heavily upon the auxiliary cost, whichis essentially a much more stable running average of the costover the most recent frames. By maintaining a reasonablyhigh value of $\gamma _ { t }$ , the auxiliary cost also preserves temporaledges, essentially reducing over-smoothing of a pixel’s dis-parity when a pixel transitions from one depth to another insubsequent frames.

![](http://127.0.0.1:9000/mineru/images/bf8f5f59a493145cbc95a8faee6647b4ec4498455e1d74926f73759bcca78fa3.jpg)  
Figure 3: Performance of temporal matching at different levelsof uniformly distributed image noise $\{ \pm 0 , \pm 2 0 , \pm 4 0 \}$ . Meansquared error (MSE) of disparities is plotted versus the valuesof the feedback coefficient $\lambda$ . Dashed lines correspond to thevalues of MSE obtained without temporal aggregation.

Table I: Parameters used in the evaluation of real-time tempo-ral stereo matching.  

<table><tr><td>Symbol</td><td>Description</td><td>Value</td></tr><tr><td></td><td>Window size for cost aggregation</td><td>33</td></tr><tr><td>T</td><td>Color difference truncation value</td><td>40</td></tr><tr><td>Yc</td><td>Strength of grouping by color similarity 1</td><td>0.03</td></tr><tr><td>Yg</td><td>Strength of grouping by proximity</td><td>0.03</td></tr><tr><td>入</td><td>Temporal feedback coefficient</td><td>varied</td></tr><tr><td>Yt</td><td>Strength of temporal grouping</td><td>0.01</td></tr><tr><td>k</td><td>Number of iterations in refinement stage</td><td>3</td></tr><tr><td>α</td><td>Disparity difference penalty</td><td>0.08</td></tr></table>

1 To enable propagation of disparity information in the iterativerefinement stage, the values of $\gamma _ { c }$ and $\gamma _ { g }$ were set to 0.09 and0.01, respectively.

![](http://127.0.0.1:9000/mineru/images/35ef673cb275b29ac2f09ff79c206dbcec10579cf7ff694213627f97b0a994d3.jpg)  
Figure 4: Optimal values of the feedback coefficient $\lambda$ cor-responding to the smallest mean squared error (MSE) of thedisparity estimates for a range of noise strengths.

![](http://127.0.0.1:9000/mineru/images/b0dc09606bd72462edeeee5e38a5e169b54cb112163998484ad794ef0df78212.jpg)  
Figure 5: A comparison of stereo matching without temporalcost aggregation (top) and with temporal cost aggregation(bottom) for a single frame in the synthetic video sequencewhere the noise is $\pm 3 0$ and the feedback coefficient is $\lambda = 0 . 8$ .

the proposed implementation achieves the highest speed ofoperation measured by the number of disparity hypothesesevaluated per second, as shown in Table II. It is also the secondmost accurate real-time method in terms of error rate, asmeasured using the Middlebury stereo evaluation benchmark.It should be noted that it is difficult to establish an unbiasedmetric for speed comparisons, as the architecture, number ofcores, and clock speed of graphics hardware used are notconsistent across implementations.

Table II: A comparison of speed and accuracy for the imple-mentations of many leading real-time stereo matching meth-ods.

<table><tr><td>Method</td><td>GPU</td><td>MDE/s1</td><td>FPS2</td><td>Error3</td></tr><tr><td>OurMethod</td><td>GeForce GTX680</td><td>215.7</td><td>90</td><td>6.20</td></tr><tr><td>CostFilter[10]</td><td>GeForceGTX480</td><td>57.9</td><td>24</td><td>5.55</td></tr><tr><td>FastBilateral[7]</td><td>Tesla C2070</td><td>50.6</td><td>21</td><td>7.31</td></tr><tr><td>RealtimeBFV[8]</td><td>GeForce 8800 GTX</td><td>114.3</td><td>46</td><td>7.65</td></tr><tr><td>RealtimeBP[21]</td><td>GeForce7900 GTX</td><td>20.9</td><td>8</td><td>7.69</td></tr><tr><td>ESAW[6]</td><td>GeForce 8800 GTX</td><td>194.8</td><td>79</td><td>8.21</td></tr><tr><td>RealTimeGPU[5]</td><td>Radeon XL1800</td><td>52.8</td><td>21</td><td>9.82</td></tr><tr><td>DCBGrid[19]</td><td>Quadro FX5800</td><td>25.1</td><td>10</td><td>10.90</td></tr></table>

1 Millions of Disparity Estimates per Second.2 Assumes $3 2 0 \times 2 4 0$ images with 32 disparity levels.3 As measured by the Middlebury stereo performance benchmark usingthe avgerage $\%$ of bad pixels.

# V. CONCLUSION

While the majority of stereo matching algorithms focuson achieving high accuracy on still images, the volume ofresearch aimed at recovery of temporally consistent disparitymaps remains disproportionally small. This paper introducesan efficient temporal cost aggregation scheme that can easilybe combined with conventional spatial cost aggregation toimprove the accuracy of stereo matching when operating onvideo sequences. A synthetic video sequence, along withground truth disparity data, was generated to evaluate theperformance of the proposed method. It was shown thattemporal aggregation is significantly more robust to noise thana method that only considers the current stereo frames.

# REFERENCES

The optimal value of the feedback coefficient is largelydependent on the noise being added to the image. Figure 4shows the optimal values of $\lambda$ for noise ranging between $\pm 0$to $\pm 4 0$ . As intuition would suggest, it is more beneficial torely on the auxiliary cost when noise is high and it is morebeneficial to rely on the current cost when noise is low. Figure5 illustrates the improvements that are achieved when applyingtemporal stereo matching to a particular pair of frames in thesynthetic video sequence. Clearly, the noise in the disparitymap is drastically reduced when temporal stereo matching isused.

The algorithm was implement using NVIDIA’s ComputeUnified Device Architecture (CUDA). The details of the im-plementation are similar to those given in [3]. When comparedto other existing real-time stereo matching implementations,

[1] D. Scharstein and R. Szeliski, “A taxonomy and evaluation of densetwo-frame stereo correspondence algorithms,” International Journal ofComputer Vision, vol. 47, pp. 7–42, April-June 2002.[2] D. Scharstein and R. Szeliski, “High-accuracy stereo depth maps usingstructured light,” in In IEEE Computer Society Conference on ComputerVision and Pattern Recognition, vol. 1, pp. 195–202, June 2003.[3] J. Kowalczuk, E. Psota, and L. Perez, “Real-time stereo matching onCUDA using an iterative refinement method for adaptive support-weightcorrespondences,” Circuits and Systems for Video Technology, IEEETransactions on, vol. 23, pp. 94 –104, Jan. 2013.[4] K.-J. Yoon and I.-S. Kweon, “Locally adaptive support-weight approachfor visual correspondence search,” in CVPR ’05: Proceedings of the 2005IEEE Computer Society Conference on Computer Vision and PatternRecognition (CVPR’05) - Volume 2, (Washington, DC, USA), pp. 924–931, IEEE Computer Society, 2005.[5] L. Wang, M. Liao, M. Gong, R. Yang, and D. Nister, “High-quality real-time stereo using adaptive cost aggregation and dynamic programming,”in 3DPVT ’06: Proceedings of the Third International Symposiumon 3D Data Processing, Visualization, and Transmission (3DPVT’06),(Washington, DC, USA), pp. 798–805, IEEE Computer Society, 2006.[6] W. Yu, T. Chen, F. Franchetti, and J. C. Hoe, “High performance stereovision designed for massively data parallel platforms,” Circuits andSystems for Video Technology, IEEE Transactions on, vol. 20, pp. 1509–1519, November 2010.[7] S. Mattoccia, M. Viti, and F. Ries, “Near real-time fast bilateral stereoon the GPU,” in Computer Vision and Pattern Recognition Workshops(CVPRW), 2011 IEEE Computer Society Conference on, pp. 136 –143,June 2011.[8] K. Zhang, J. Lu, Q. Yang, G. Lafruit, R. Lauwereins, and L. Van Gool,“Real-time and accurate stereo: A scalable approach with bitwise fastvoting on CUDA,” Circuits and Systems for Video Technology, IEEETransactions on, vol. 21, pp. 867 –878, July 2011.[9] C. Rhemann, A. Hosni, M. Bleyer, C. Rother, and M. Gelautz, “Fast cost-volume filtering for visual correspondence and beyond,” in ComputerVision and Pattern Recognition (CVPR), 2011 IEEE Conference on,pp. 3017 –3024, June 2011.[10] A. Hosni, C. Rhemann, M. Bleyer, and M. Gelautz, “Temporally con-sistent disparity and optical flow via efficient spatio-temporal filtering,”in Advances in Image and Video Technology (Y.-S. Ho, ed.), vol. 7087of Lecture Notes in Computer Science, pp. 165–177, Springer Berlin /Heidelberg, 2012.[11] C. Tomasi and R. Manduchi, “Bilateral filtering for gray and colorimages,” in Computer Vision, 1998. Sixth International Conference on,pp. 839 –846, jan 1998.[12] K. He, J. Sun, and X. Tang, “Guided image filtering,” in ComputerVision – ECCV 2010, vol. 6311 of Lecture Notes in Computer Science,pp. 1–14, Springer Berlin / Heidelberg, 2010.[13] L. Zhang, B. Curless, and S. M. Seitz, “Spacetime stereo: Shaperecovery for dynamic scenes,” in IEEE Computer Society Conferenceon Computer Vision and Pattern Recognition, pp. 367–374, June 2003.[14] J. Davis, D. Nehab, R. Ramamoorthi, and S. Rusinkiewicz, “Spacetimestereo: a unifying framework for depth from triangulation,” PatternAnalysis and Machine Intelligence, IEEE Transactions on, vol. 27,pp. 296 –302, February 2005.[15] E. Larsen, P. Mordohai, M. Pollefeys, and H. Fuchs, “Temporallyconsistent reconstruction from multiple video streams using enhancedbelief propagation,” in Computer Vision, 2007. ICCV 2007. IEEE 11thInternational Conference on, pp. 1 –8, oct. 2007.[16] M. Bleyer, M. Gelautz, C. Rother, and C. Rhemann, “A stereo approachthat handles the matting problem via image warping,” in ComputerVision and Pattern Recognition, 2009. CVPR 2009. IEEE Conferenceon, pp. 501 –508, June 2009.[17] M. Sizintsev and R. Wildes, “Spatiotemporal stereo via spatiotemporalquadric element (stequel) matching,” in Computer Vision and PatternRecognition, 2009. CVPR 2009. IEEE Conference on, pp. 493 –500,june 2009.[18] M. Sizintsev and R. Wildes, “Spatiotemporal stereo and scene flow viastequel matching,” Pattern Analysis and Machine Intelligence, IEEETransactions on, vol. 34, pp. 1206 –1219, june 2012.[19] C. Richardt, D. Orr, I. Davies, A. Criminisi, and N. A. Dodgson,“Real-time spatiotemporal stereo matching using the dual-cross-bilateralgrid,” in Proceedings of the European Conference on Computer Vision(ECCV), Lecture Notes in Computer Science, pp. 510–523, September2010.[20] S. Paris and F. Durand, “A fast approximation of the bilateral filter usinga signal processing approach,” Int. J. Comput. Vision, vol. 81, pp. 24–52,Jan. 2009.[21] Q. Yang, L. Wang, R. Yang, S. Wang, M. Liao, and D. Nistér, “Real-time global stereo matching using hierarchical belief propagation.,” inBritish Machine Vision Conference, pp. 989–998, 2006.