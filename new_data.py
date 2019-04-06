from numpy import *
import pandas as pd
# h = w1 + w2 * x

def compute_error_for_line_given_points(b,m,c,d,e,f,g,h,points):
	totalError = 0
	for i in range(1,30):
		x1 = float(points.at[i,1])
		x2 = float(points.at[i,2])
		x3 = float(points.at[i,3])
		x4 = float(points.at[i,4])
		x5 = float(points.at[i,5])
		x6 = float(points.at[i,6])
		x7 = float(points.at[i,7])
		y = float(points.at[i,8])
		totalError += ((y - ((h*x7)+(g*x6)+(f*x5)+(e*x4)+
									(d*x3)+(c*x2)+(m*x1)+b))**2)
	return totalError / float(len(points))

def step_gradient(b_current,m_current,c_current,d_current,e_current,
					f_current,g_current,h_current,points,learningRate):
	b_gradient = 0
	m_gradient = 0
	c_gradient = 0
	d_gradient = 0
	e_gradient = 0
	f_gradient = 0
	g_gradient = 0
	h_gradient = 0
	N = float(len(points))
	for i in range(1,30):
		x1 = float(points.at[i,1])
		x2 = float(points.at[i,2])
		x3 = float(points.at[i,3])
		x4 = float(points.at[i,4])
		x5 = float(points.at[i,5])
		x6 = float(points.at[i,6])
		x7 = float(points.at[i,7])
		y = float(points.at[i,8])
		b_gradient += -(1/N) * (y -((h_current*x7)+
				(g_current*x6)+(f_current*x5)+(e_current*x4)+
				(d_current*x3)+(c_current*x2)+(m_current*x1) + b_current))
		m_gradient += -(1/N) * x1 * (y -((h_current*x7)+
				(g_current*x6)+(f_current*x5)+(e_current*x4)+
				(d_current*x3)+(c_current*x2)+(m_current*x1) + b_current))
		c_gradient = -(1/N) * x2 * (y -((h_current*x7)+
				(g_current*x6)+(f_current*x5)+(e_current*x4)+
				(d_current*x3)+(c_current*x2)+(m_current*x1) + b_current))
		d_gradient = -(1/N) * x3 * (y -((h_current*x7)+
				(g_current*x6)+(f_current*x5)+(e_current*x4)+
				(d_current*x3)+(c_current*x2)+(m_current*x1) + b_current))
		e_gradient = -(1/N) * x4 * (y -((h_current*x7)+
				(g_current*x6)+(f_current*x5)+(e_current*x4)+
				(d_current*x3)+(c_current*x2)+(m_current*x1) + b_current))
		f_gradient = -(1/N) * x5 * (y -((h_current*x7)+
				(g_current*x6)+(f_current*x5)+(e_current*x4)+
				(d_current*x3)+(c_current*x2)+(m_current*x1) + b_current))
		g_gradient = -(1/N) * x6 * (y -((h_current*x7)+
				(g_current*x6)+(f_current*x5)+(e_current*x4)+
				(d_current*x3)+(c_current*x2)+(m_current*x1) + b_current))
		h_gradient = -(1/N) * x7 * (y -((h_current*x7)+
				(g_current*x6)+(f_current*x5)+(e_current*x4)+
				(d_current*x3)+(c_current*x2)+(m_current*x1) + b_current))
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	new_c = c_current - (learningRate * c_gradient)
	new_d = d_current - (learningRate * d_gradient)
	new_e = e_current - (learningRate * e_gradient)
	new_f = f_current - (learningRate * f_gradient)
	new_g = g_current - (learningRate * g_gradient)
	new_h = h_current - (learningRate * h_gradient)
	return new_b,new_m,new_c,new_d,new_e,new_f,new_g,new_h

def gradient_descent_runner(points,starting_b,starting_m,starting_c,
	starting_d,starting_e,starting_f,starting_g,starting_h,learning_rate,
												num_iterations):
	b = starting_b
	m = starting_m
	c = starting_c
	d = starting_d
	e = starting_e
	f = starting_f
	g = starting_g
	h = starting_h
	for i in range(num_iterations):
		b,m,c,d,e,f,g,h = step_gradient(b,m,c,d,e,f,g,h,points,learning_rate)
	return b,m,c,d,e,f,g,h

def run():
	points = pd.read_csv("Admission_Predict.csv",sep=",",header=None)
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	initial_c = 0
	initial_d = 0
	initial_e = 0
	initial_f = 0
	initial_g = 0
	initial_h = 0
	num_iterations = 1000
	print("Starting gradient descent at b = {0}, m = {1},c = {3},d = {4} error = {2}".format(initial_b, initial_m, 
		compute_error_for_line_given_points(initial_b, initial_m,initial_c,initial_d,initial_e,initial_f,initial_g,initial_h,points),initial_c,initial_d))
	print('Running...')
	[b, m, c, d, e, f, g, h] = gradient_descent_runner(points, initial_b, initial_m,initial_c,initial_d,initial_e,initial_f,initial_g,initial_h, learning_rate, num_iterations)
	print("After {0} iterations b = {1}, m = {2},c = {4},d = {5} error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m,c,d,e,f,g,h, points),c,d))
	predict = b + m*324 + c*107 + d*4 + e*4 + f*4.5 + g*8.87 + h*1
	print(predict)
if __name__ == '__main__':
	run()

# Все работает!