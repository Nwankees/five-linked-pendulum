# Understanding the project

## One control step

Imagine pausing the animation. There are twelve measurements: cart position,
cart speed, five link angles and five link angular speeds. They appear in that
order in `environment.states`.

`controller.action(states, step)` asks the trained network for a number. An
action of 1 means 45 newtons to the right; -1 means 45 newtons to the left.
`environment.step(action)` holds that force for 0.02 seconds, advancing the
physics four times by 0.005 seconds. Then the next decision is made.

The first dimension in `states` identifies the cart. Normally the animation
has one cart. Evaluations can run 512 independent carts together. A statement
like `states[:, 2 + link]` means "this link's angle in every cart". The link
loop is still explicit; NumPy only performs the same calculation for all carts.

## Why the physics needs a matrix

Push an isolated cart: force divided by mass gives acceleration. Attach a
pendulum: some force also accelerates the pendulum, and its motion pushes
back on the cart. With five links, every moving part affects the others.

`physics.accelerations` writes six force-balance equations, one for the cart
and one for each link. The six unknowns are their accelerations. The mass
matrix stores the coefficient of each acceleration in each equation. The
right-hand side contains the applied force, gravity, friction and forces
arising from existing angular motion.

`np.linalg.solve(matrix, right_side)` solves those six simultaneous equations.
It does not choose the controller's force. The controller has already chosen it.
The double `for` loop fills the link-to-link entries explicitly.

`mass_at_or_above(link)` adds the masses supported by a link. A lower link
must move the masses attached above it, too. `sin` and `cos` convert the
angles into horizontal and vertical effects. Squared angular speed appears
because rotational motion produces centripetal acceleration.

## Why estimate the motion four times?

Velocities and accelerations change during a time step. RK4 first predicts
the slope at the start, twice near the middle, and once near the end. It
uses their weighted average to advance the state. The four estimates reduce
numerical error; they are not four separate control decisions.

## Checking balance with a loop

Start with each cart provisionally balanced. For every link, check its angle
and speed. A failed check clears that cart's balanced flag. Only carts with
all ten checks passing accumulate hold time. If any link fails a later check,
that continuous hold restarts from zero.

Success remains recorded once a two-second hold happens, but evaluation also
requires `not failed`: exiting the track later invalidates the episode.

## Learning an action from an example

Each demonstration records a state and the expert's chosen force. For one
route and one stage, training finds the typical state, subtracts it from each
example, and rescales each measurement. This keeps very different units and
variances from overwhelming one another.

The hidden layer splits a number into two parts: `ReLU(x)` and `ReLU(-x)`.
For example, -3 becomes 0 and 3. Subtracting those outputs can recover the
original number, so the network can represent either sign of a correction.
The output layer learns how much each part should contribute to the force.

`torch.linalg.lstsq` chooses output weights that minimize squared mistakes
on the example forces. The extra column of ones learns the bias, which is
the force predicted at the typical state. The small regularization discourages
extreme weights when examples contain little information about a direction.

This is a constrained neural-network training method. It intentionally does
not learn the hidden layer. The ordinary MLP experiment does learn both hidden
layers with Adam; that simpler universal-policy attempt failed its rollout test.

## Choosing the demonstration route

At reset, the controller compares the starting state to the eight demonstrated
starting states. It loops over every state variable, adds squared scaled
differences, and keeps the route with the smallest total. The angle difference
is wrapped so +179 degrees and -179 degrees count as two degrees apart.

Each subsequent step uses that route's next network. The elapsed step is
therefore part of the policy, even though it is not a network input. Once
the 250 swing-up steps end, the balance network acts on the measured state.

## What the tests establish

The physics tests check equilibrium, force coupling, numerical convergence,
reflection symmetry and energy accounting, as well as episode rules. The
learning tests check that training actually fits labelled force examples and
that a saved policy gives the same output when reloaded.

The evaluation reports establish performance on the stated downward-start
distribution. They do not prove all possible initial states are controlled,
that large manual disturbances are recoverable, or that a real mechanism
would work without further modelling and testing.
