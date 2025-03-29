
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
#ideas
    #consider adding a pareto dimension dealing with unique coverage of instances (i.e. so if there are rules that cover instances that no other rule covers there is incentive to preserve them)
    # We can adjust the importance of either objective by adjusting the max scale of each objective (currently equal from 0 to 1)
class PARETO:
    def __init__(self):  
        #Definte the parts of the Pareto Front
        self.bin_front = []  #list of objective pair sets (log_rank, area) for each non-dominated rule (ordered by increasing accuracy)
        self.bin_front_scaled = []
        self.max_score = None
        self.max_area = None
        self.front_diagonal_lengths = [] # length of the diagonal lines from the orgin to each point on the pareto front (used to calculate rule fitness)

    def update_front(self,candidate_metric_1,candidate_metric_2,objectives):
        """  Handles process of checking and updating the rule pareto front. Only set up for two objectives. """
        original_front = copy.deepcopy(self.bin_front)
        candidate_bin = (candidate_metric_1,candidate_metric_2)
        if not candidate_bin in self.bin_front: # Check that candidate rule objectives are not equal to any existing objective pairs on the front
            non_dominated_bins = []
            candidate_dominated = False
            for front_bin in self.bin_front:
                if self.dominates(front_bin,candidate_bin,objectives): #does front rule dominate candidate rule (if so, this check is ended)
                    candidate_dominated = True #prevents candidate from being added 
                    break # original front is preserved
                elif not self.dominates(candidate_bin,front_bin,objectives): #does the candidate rule dominate the front rule (if so it will get added to the front)
                    non_dominated_bins.append(front_bin)
            if candidate_dominated: #at least one front rule dominates the candidate rule
                non_dominated_bins = self.bin_front
            else: #no front rules dominate the candidate rule
                non_dominated_bins.append(candidate_bin)
            # Update the rule front to include only non dominated rules, sort by log rank score
            self.bin_front = sorted(non_dominated_bins, key=lambda x: x[0])

            # Update the maximum log rank and low risk area in the pareto front
            self.max_score = max(self.bin_front, key=lambda x: x[0])[0]
            self.max_area = max(self.bin_front, key=lambda x: x[1])[1]
            # if max score and area have been found, scale the bin
            if self.max_score != None and self.max_area != None:
                self.bin_front_scaled = [(x[0] / float(self.max_score), x[1] / float(self.max_area)) for x in self.bin_front]
            else:
                self.bin_front_scaled = self.bin_front

        if original_front == self.bin_front:
            return False
        else:
            return True
      
    def delete_from_front(self, candidate_metric_1, candidate_metric_2):
        bin = (candidate_metric_1, candidate_metric_2)
        bin_scaled = (candidate_metric_1/self.max_score, candidate_metric_2/self.max_area)
        self.bin_front.remove(bin)
        self.bin_front_scaled.remove(bin_scaled)

        # update max score and area in case a max bin was deleted
        if candidate_metric_1 == self.max_area or candidate_metric_2 == self.max_score:
            self.max_score = max(self.bin_front, key=lambda x: x[0])[0]
            self.max_area = max(self.bin_front, key=lambda x: x[1])[1]

    def dominates(self,p,q,objectives):
        """Check if p dominates q. A rule dominates another if it has a more optimal value for at least one objective."""
        better_in_all_objectives = True
        better_in_at_least_one_objective = False
        for val1, val2, obj in zip(p, q, objectives):
            if obj == 'max':
                if val1 < val2:
                    better_in_all_objectives = False
                if val1 > val2:
                    better_in_at_least_one_objective = True
            elif obj == 'min':
                if val1 > val2:
                    better_in_all_objectives = False
                if val1 < val2:
                    better_in_at_least_one_objective = True
            else:
                raise ValueError("Objectives must be 'max' or 'min'")
        return better_in_all_objectives and better_in_at_least_one_objective

    def get_pareto_fitness(self,log_rank_score,low_risk_area, landscape):
        """ Calculate rule fitness releative to the rule pareto front. Only set up for two objectives. """
        scaled_area = low_risk_area / self.max_area
        scaled_score = log_rank_score / self.max_score
        # First handle simple special cases
        if len(self.bin_front_scaled) == 1 and self.bin_front_scaled[0][0] == 0.0 and self.bin_front_scaled[0][1] == 0.0:
            return None
        if landscape: # Handles special cases when calculating fitness landscape for visualization
            if scaled_score > 1.0 or scaled_area > 1.0:
                return 1.0
            if self.point_beyond_front(scaled_score, scaled_area):
                return 1.0 
        if log_rank_score == 0.0 and low_risk_area == 0.0:
            return 0.0
        elif (log_rank_score,low_risk_area) in self.bin_front: # Rules on the front return an ideal fitness
            return 1.0
        elif log_rank_score >= self.max_score:
            return 1.0
        elif low_risk_area >= self.max_area:
            return 1.0
        #elif rule_coverage == self.metric_limits[1]: #rule has the maximum value of one objective
        #    return 1.0
        else:
            bin_objectives = (scaled_score, scaled_area)
            # Find the closest distance between the bin and the pareto front
            temp_front = [(0.0, self.max_area)] #max area boundary
            for front_point in self.bin_front_scaled:
                temp_front.append(front_point)
            temp_front.append([self.max_score, 0.0]) #max area boundary
            min_distance = float('inf')
            for i in range(len(temp_front) - 1):
                segment_start = temp_front[i]
                segment_end = temp_front[i + 1]
                distance = self.point_to_segment_distance(bin_objectives, segment_start, segment_end)
                min_distance = min(min_distance, distance)
            pareto_fitness = 1 - min_distance
            return pareto_fitness
        
    def point_to_segment_distance(self, point, segment_start, segment_end):
        """ """
        # Vector from segment start to segment end (normalize vector so starting point is 0,0)
        segment_vector = np.array(segment_end) - np.array(segment_start)
        # Vector from segment start to the point (normalize vector so starting point is 0,0)
        point_vector = np.array(point) - np.array(segment_start)
        # Project the point_vector onto the segment_vector to find the closest point on the segment
        segment_length_squared = np.dot(segment_vector, segment_vector) #Calculate segment length squared to be used to do the projection
        if segment_length_squared == 0: #Safty check that the segment start and stop are not the same (if so just return distance to that single point)
            # The segment start and end points are the same
            return self.euclidean_distance(point, segment_start)
        # If segment has length: project the point vector onto the segment vector (identifies the perpendicular intersect)
        projection = np.dot(point_vector, segment_vector) / segment_length_squared
        projection_clamped = max(0, min(1, projection)) #checks if the interstect is within the segment (because the projection was forced between 0, and 1)
        # Find the closest point on the segment (either the perpendicular intersect or distance to segment end)
        closest_point_on_segment = np.array(segment_start) + projection_clamped * segment_vector
        # Return the distance from the point to this closest point on the segment
        return self.euclidean_distance(point, closest_point_on_segment)
    
    def euclidean_distance(self,point1,point2):
        """ Calculates the euclidean distance between two n-dimensional points"""
        if len(point1) != len(point2):
            raise ValueError("Both points must have the same number of dimensions")
        distance = math.sqrt(sum((y - x) ** 2 for y, x in zip(point1, point2)))
        return distance

    def slope(self,point1,point2):
        """ Calculates the slopes between two 2-dimensional points """
        if point1[1] == point2[1]: # line is vertical (both points have 0 coverage)
            slope = np.inf
        else:
            slope = (point2[0] - point1[0]) / (point2[1] - point1[1])
        return slope
    
    def point_beyond_front(self,scaled_score,scaled_area):
        """ Used for creating pareto front landscape visualization background fitness landscape. """
        # Define line segment from the origin (0,0) to the rule's objective (x,x)
        bin_start = (0,0)
        bin_end = (scaled_score, scaled_area)
        # Identify segments making up front to check
        intersects = False
        i = 0
        while not intersects and i < len(self.bin_front) - 1:
            segment_start = self.bin_front_scaled[i]
            segment_end = self.bin_front_scaled[i + 1]
            intersects = self.do_intersect(bin_start,bin_end,segment_start,segment_end)
            if intersects:
                return True
            i += 1
        return False
    
    def do_intersect(self,p1, q1, p2, q2):
        """ Main function to check whether the line segment p1q1 and p2q2 intersect. """
        # Find the four orientations needed for the general and special cases
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
        # General case
        if o1 != o2 and o3 != o4:
            return True
        # Special cases
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True
        # p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True
        # p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True
        # p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True
        return False

    def on_segment(self, p, q, r):
        """
        Given three collinear points p, q, r, the function checks if point q lies on the segment pr.
        """
        if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
            return True
        return False

    def orientation(self, p, q, r):
        """
        To find the orientation of the ordered triplet (p, q, r).
        The function returns:
        0 -> p, q, and r are collinear
        1 -> Clockwise
        2 -> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2
        
    def plot_zoomed_pareto_landscape(self, resolution, min_area, bin_pop, show, save, output_path, data_name):
        # Generate fitness landscape ******************************
        x = np.linspace(0.995 * min_area, 1.005 * self.max_area,resolution) #area
        y = np.linspace(0,1.025 * self.max_score,resolution) #log rank
        #X,Y = np.meshgrid(x,y)
        Z = [[None for _ in range(resolution)] for _ in range(resolution)]
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j][i] = self.get_pareto_fitness(y[j], x[i], True) #log rank,area  (rows,columns)
        # Prepare to plot rule front *****************************
        metric_1_front_list = [None]*len(self.bin_front_scaled)
        metric_2_front_list = [None]*len(self.bin_front_scaled)
        i = 0
        for bin in self.bin_front:
          metric_1_front_list[i] = bin[0]
          metric_2_front_list[i] = bin[1]
          i+=1
        # Plot Setup *********************************************
        plt.figure(figsize=(10,6)) #(10, 8))
        im = plt.imshow(Z, extent=[0.995 * min_area, 1.005*self.max_area, 0, 1.025*self.max_score], interpolation='nearest', origin='lower', cmap='magma', aspect='auto') #cmap='viridis' 'magma', alpha=0.6
        # Add colorbar for the gradient
        cbar = plt.colorbar(im)
        cbar.set_label('Fitness Value')

        # Plot rule front ***************************************
        plt.plot(np.array(metric_2_front_list), np.array(metric_1_front_list), 'o-', ms=10, lw=2, color='black')
        # Plot pareto front boundaries to plot edge
        plt.plot([metric_2_front_list[-1],0],[metric_1_front_list[-1],metric_1_front_list[-1]],'--',lw=1, color='black') # Accuracy line
        plt.plot([metric_2_front_list[0],metric_2_front_list[0]],[metric_1_front_list[0],0],'--',lw=1, color='black') # Accuracy line
        # Add labels and title
        plt.xlabel('Low Risk Area', fontsize=14)
        plt.ylabel('Log Rank Score', fontsize=14)
        # Set the axis limits between 0 and 1
        plt.xlim(0.995 * min_area, 1.005*self.max_area)
        plt.ylim(0, 1.025*self.max_score)
        custom_xticks = np.around(np.linspace(0.995 * min_area, 1.005*self.max_area, 10), 3)
        custom_xlabels = np.around(np.linspace(0.995 * min_area, 1.005*self.max_area, 10), 3)
        plt.xticks(custom_xticks, custom_xlabels)

        custom_yticks = np.around(np.linspace(0, 1.025*self.max_score, 10), 1)
        custom_ylabels = np.around(np.linspace(0, 1.025*self.max_score, 10), 1)
        plt.yticks(custom_yticks, custom_ylabels)
        # Prepare to plot rule population ***********************
        master_metric_1_list = []
        master_metric_2_list = []
        for i in range(len(bin_pop)):
            bin = bin_pop[i]
            master_metric_1_list.append(bin.log_rank_score)
            master_metric_2_list.append(bin.low_risk_area)
            
        plt.plot(np.array(master_metric_2_list), np.array(master_metric_1_list), 'o', ms=3, lw=1, color='grey')
        plt.legend(loc='upper left', bbox_to_anchor=(1.25, 1), fontsize='small')
        plt.subplots_adjust(right=0.75)
        if save:
            plt.savefig(output_path+'/'+data_name+'_pareto_fitness_landscape.png', bbox_inches="tight")
        if show:
            plt.show()
    
    def plot_pareto_landscape(self, resolution, min_area, bin_pop, show, save, output_path, data_name):
        # Generate fitness landscape ******************************
        x = np.linspace(0, 2,resolution) #area
        y = np.linspace(0, 1.025 * self.max_score,resolution) #log rank
        #X,Y = np.meshgrid(x,y)
        Z = [[None for _ in range(resolution)] for _ in range(resolution)]
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j][i] = self.get_pareto_fitness(y[j], x[i], True) #log rank,area  (rows,columns)
        # Prepare to plot rule front *****************************
        metric_1_front_list = [None]*len(self.bin_front_scaled)
        metric_2_front_list = [None]*len(self.bin_front_scaled)
        i = 0
        for bin in self.bin_front:
          metric_1_front_list[i] = bin[0]
          metric_2_front_list[i] = bin[1]
          i+=1
        # Plot Setup *********************************************
        plt.figure(figsize=(10,6)) #(10, 8))
        im = plt.imshow(Z, extent=[0, 2, 0, 1.025*self.max_score], interpolation='nearest', origin='lower', cmap='magma', aspect='auto') #cmap='viridis' 'magma', alpha=0.6
        # Add colorbar for the gradient
        cbar = plt.colorbar(im)
        cbar.set_label('Fitness Value')

        # Plot rule front ***************************************
        plt.plot(np.array(metric_2_front_list), np.array(metric_1_front_list), 'o-', ms=10, lw=2, color='black')
        # Plot pareto front boundaries to plot edge
        plt.plot([metric_2_front_list[-1],0],[metric_1_front_list[-1],metric_1_front_list[-1]],'--',lw=1, color='black') # Accuracy line
        plt.plot([metric_2_front_list[0],metric_2_front_list[0]],[metric_1_front_list[0],0],'--',lw=1, color='black') # Accuracy line
        # Add labels and title
        plt.xlabel('Low Risk Area', fontsize=14)
        plt.ylabel('Log Rank Score', fontsize=14)
        # Set the axis limits between 0 and 1
        plt.xlim(0, 2)
        plt.ylim(0, 1.025*self.max_score)
        custom_xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        custom_xlabels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        plt.xticks(custom_xticks, custom_xlabels)

        custom_yticks = np.around(np.linspace(0, 1.025*self.max_score, 10), 1)
        custom_ylabels = np.around(np.linspace(0, 1.025*self.max_score, 10), 1)
        plt.yticks(custom_yticks, custom_ylabels)
        # Prepare to plot rule population ***********************
        master_metric_1_list = []
        master_metric_2_list = []
        for i in range(len(bin_pop)):
            bin = bin_pop[i]
            master_metric_1_list.append(bin.log_rank_score)
            master_metric_2_list.append(bin.low_risk_area)
            
        plt.plot(np.array(master_metric_2_list), np.array(master_metric_1_list), 'o', ms=3, lw=1, color='grey')
        plt.legend(loc='upper left', bbox_to_anchor=(1.25, 1), fontsize='small')
        plt.subplots_adjust(right=0.75)
        if save:
            plt.savefig(output_path+'/'+data_name+'_pareto_fitness_landscape.png', bbox_inches="tight")
        if show:
            plt.show()

    """
    def get_pareto_fitness_old(self,rule_accuracy,rule_coverage):
        # First handle simple special cases
        if len(self.bin_front_scaled) == 1 and self.bin_front_scaled[0][0] == 0.0 and self.bin_front_scaled[0][1] == 0.0:
            return None
        elif rule_accuracy == 0.0 and rule_coverage == 0.0:
            return 0.0
        elif (rule_accuracy,rule_coverage) in self.bin_front: # Rules on the front return an ideal fitness
            return 1.0
        elif rule_accuracy == self.metric_limits[0]:
            return 1.0
        #elif rule_coverage == self.metric_limits[1]: #rule has the maximum value of one objective
        #    return 1.0
        else:
            scaled_rule_coverage = rule_coverage / float(self.metric_limits[1])
            rule_objectives = (rule_accuracy,scaled_rule_coverage)
            # Calculate distance from pareto front origin (0,0) to the rule_objectives on the pareto front
            rule_distance = self.euclidean_distance((0.0,0.0),rule_objectives)
            # Calculate slope between pareto front origin (0,0) and the rule_objectives on the pareto front
            rule_slope = self.slope((0.0,0.0),rule_objectives)
            # Calculate distance from origin to pareto front line segment intercept based on rule_slope
            line_coordinates = []
            segment_index = None
            if len(self.bin_front_scaled) == 1: #front is comprised of a single rule
                # Identify the line segment (coordinates) of the pareto front where the target rule's slope will intersect the front.
                front_point_slope = self.slope((0.0,0.0),self.bin_front_scaled[0])
                if front_point_slope < rule_slope:
                    line_coordinates = [self.bin_front_scaled[0],(self.bin_front_scaled[0][0],0.0)] #defines max accuracy horizontal line
                elif front_point_slope > rule_slope:
                    segment_index = 0
                    line_coordinates = [(0.0,self.bin_front_scaled[0][1]),self.bin_front_scaled[0]] #defines max coverage vertical line
                else:
                    line_coordinates = None
            else: #front is comprised of multiple rules
                # Identify the line segment of the pareto front where the target rule's slope will intersect the front.
                segment_index = 0 #starts with edge vertical line (max coverage)
                front_point_slope = 0.0 #starts with horizontal line to maximum coverage
                while rule_slope >= front_point_slope and segment_index < len(self.bin_front_scaled): #starts with rule on front with maximum coverage
                    #if segment_index <= len(self.bin_front_scaled)-1:
                    front_point_slope = self.slope((0.0,0.0),self.bin_front_scaled[segment_index])
                    segment_index +=1 

                #print(segment_index)
                # Identify the line segment (coordinates) of the pareto front where the target rule's slope will intersect the front.
                if segment_index == 0: #defines max coverage vertical line
                    line_coordinates = [(0.0,self.bin_front_scaled[0][1]),self.bin_front_scaled[0]] #defines max coverage vertical line
                elif segment_index == len(self.bin_front_scaled): #defines max accuracy horizontal line
                    line_coordinates = [self.bin_front_scaled[0],(self.bin_front_scaled[0][0],0.0)] #defines max accuracy horizontal line
                else:
                    line_coordinates = [self.bin_front_scaled[segment_index-1],self.bin_front_scaled[segment_index]]
            # Given the pair of front line coordinates and the slope of the rule line, identify the intercept point
            if line_coordinates != None:
                intercept = self.find_line_intercept(line_coordinates[0],line_coordinates[1],rule_slope)
            else:
                intercept = self.bin_front_scaled[0]
            # Calculate the distance from the origin to that intercept
            front_distance = self.euclidean_distance((0.0,0.0),intercept)
            # Use the relative distance between the origin and the intercept to calculate the rule fitness based on how close it is to the pareto front
            pareto_fitness = rule_distance / float(front_distance)
            # Adjust this calculation to account for the different distances between the origin and points on the pareto front.
            #pareto_fitness = pareto_pre_fitness * front_distance / max(self.front_diagonal_lengths)
            if segment_index != None and segment_index == 0: # If rule is on intersect with max coverage line
                pareto_fitness = pareto_fitness / float(self.heros.nu) #arbitrary penalty applied
            if pareto_fitness > 1.0:
                pareto_fitness = 1.0
            return pareto_fitness

    def find_line_intercept(self,point1,point2,rule_slope):
        # Assumes point1 has the smaller accuracy
        line_slope = self.slope(point1,point2)
        if line_slope == 0.0:
            if rule_slope == np.inf:
                yIntercept = point1[0] 
                xIntercept = 0.0
            else:
                yIntercept = point1[0] 
                xIntercept = yIntercept/rule_slope
        elif line_slope == np.inf:
            if rule_slope == 0.0:
                xIntercept = point1[1] 
                yIntercept = 0.0
            else:
                xIntercept = point1[1]
                yIntercept = xIntercept*rule_slope
        else:
            xIntercept = (point2[0] - line_slope*point2[1]) / float(rule_slope-line_slope)
            yIntercept = rule_slope*xIntercept
        return (yIntercept,xIntercept)
    """
