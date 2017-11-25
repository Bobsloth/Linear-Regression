from numpy import *

def compute_error_for_line_given_points(b, m, points):

	#initialize it at 0
	totalError = 0
	#for every point
	for i in range(0, len(points)):
		#get the x value
		x = points[i, 0]
		#get the y value
		y = points[i, 1]
		#get the difference, square it and add it to the total
		totalError += (y - (m*x+b)) ** 2
	#get the average
	return totalError/float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iteration):

	#starting b and m
	m = starting_m
	b = starting_b

	#gradient descent
	for i in range(num_iteration):
		#update m and b with the new more accurate m and b perfoming gradient descent
		b, m = step_gradient(b, m, array(points), learning_rate)


	return [b,m]

def step_gradient(current_b, current_m, points, learning_rate):

	#starting points for the gradient
	m_gradient = 0
	b_gradient = 0

	N = float(len(points))

	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]

		b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
		m_gradient += -(2/N) * x*(y - ((current_m * x) + current_b))

	#update b and m using our partial derivatives
	new_b = current_b - (learning_rate * b_gradient)
	new_m = current_m - (learning_rate * m_gradient)

	return [new_b, new_m]



def run():
	
	#Step 1: collect the data
	points = genfromtxt('data.csv', delimiter=',')
	

	#Step 2: define the hyperparameters
	#how fast should the model converge?
	learning_rate = 0.0001
	
	#y = mx + b
	initial_m = 0
	initial_b = 0
	num_iteration = 1000

	#Step 3 : train our model
	print 'starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iteration)

	
	print 'ending point at b = {1}, m = {2}, error = {3}'.format(num_iteration, b, m, compute_error_for_line_given_points(b, m, points))







if __name__ == '__main__':
	run()
