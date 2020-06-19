.. Fault Tolerance Library documentation Code Formatting file
.. _codeformatting:

Code Formatting 
======================

Code Checkers
----------------------

To enhance the code quality of FTI, we use the following open source code checkers:

.. list-table::
   :header-rows: 1

   * - Language
     - Code Checker
   * - C
     - cpplint_
   * - Fortran
     - fprettify_
   * - CMake
     - cmakelint_

.. _cpplint: https://github.com/cpplint/cpplint
.. _fprettify: https://github.com/pseewald/fprettify
.. _cmakelint: https://github.com/cheshirekow/cmake_format


Coding Style
-----------------------

cpplint checks C/C++ files style issues following Google C/C++ style guide_. Please visit this guide to understand how you should format your code to comply to FTI's style. 

Fortran's and CMake style checkers have a plenty of formatting options, as the respective documentation lists. For FTI, we choose to adopt the following style rules: 

**Formatting options for Fortran Files**

.. list-table::
   :header-rows: 1

   * - Options
     - Explanation
   * - --indent 4 
     - relative indentation width
   * - --line-length 140          
     - column after which a line should end
   * - --whitespace 2
     - Presets for the amount of whitespace : 2
   * - --strict-indent               
     - strictly impose indentation even for nested loops

**Formatting options for CMake Files**

.. list-table::
   :header-rows: 1

   * - Options
     - Explanation
   * -  --line-width 80 
     - How wide to allow formatted cmake files
   * - --tab-size 4             
     - How many spaces to tab for indent
   * - --separate-ctrl-name-with-space
     - separate flow control names from their parentheses with a space
   * - --separate-fn-name-with-space            
     - separate function names from parentheses with a space
                     


.. _guide: http://google.github.io/styleguide/cppguide.html


Implementation
----------------------

Code checking is integrated in FTI through a script that traverses any added/modified code in FTI and checks if it conforms to the desired coding style. The script acts as a pre-commit hook that gets fired by a local commit. 

Examples of the execution on FTI s code

.. image:: _static/pre-commit-fails.png
   :width: 600px
   :height: 600px

Contributing
----------------------

**Prerequisites**

Before you will be able to contribute to FTI, you need to have the code checkers installed so that your code is checked prior to any commit.
The checkers are easy to install if you have pip. For the latest installation steps, please visit the :ref:`Code Checkers`. 

..

	To make use of the pre-commit hook, after cloning the repository, one should initialize their branch through ``git init`` command.

..

	This should port the pre-commit hook, along with the default git hooks, to your ``GIT_DIR``



.. note::
	Notice: For a temporary commit where the developer is aware that the code might still need formatting but still wants to commit, use the flag **--no-verify** along with the commit command.
