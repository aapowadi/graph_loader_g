#pragma once


// stl
#include <iostream>
#include <string>
#include <functional>
#include <memory>

using namespace std;
using namespace std::placeholders;


class RV10Demo {

public:

	/*!
	Start the demon application
	*/
	static bool StartDemo(string workingDirectory = ".");

	static void Update(void);
};