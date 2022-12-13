"""
Implementation of the Circles FLAME GPU model in python, as an example.
"""

# Import pyflamegpu
import pyflamegpu
# Import standard python libs that are used
import sys, random, math

# Define FLAME GPU Agent functions as strings, for compilation via nvrtc

# Agent Function to output the agents ID and position in to a 3D spatial message list
output_message = r"""
FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    return flamegpu::ALIVE;
}
"""

# Agent function to iterate messages, and move according to the rules of the circle model
move = r"""
FLAMEGPU_AGENT_FUNCTION(move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const flamegpu::id_t ID = FLAMEGPU->getID();
    const float REPULSE_FACTOR = FLAMEGPU->environment.getProperty<float>("repulse");
    const float RADIUS = FLAMEGPU->message_in.radius();
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in(x1, y1, z1)) {
        if (message.getVariable<flamegpu::id_t>("id") != ID) {
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");
            const float z2 = message.getVariable<float>("z");
            float x21 = x2 - x1;
            float y21 = y2 - y1;
            float z21 = z2 - z1;
            const float separation = cbrt(x21*x21 + y21*y21 + z21*z21);
            if (separation < RADIUS && separation > 0.0f) {
                float k = sinf((separation / RADIUS)*3.141*-2)*REPULSE_FACTOR;
                // Normalise without recalculating separation
                x21 /= separation;
                y21 /= separation;
                z21 /= separation;
                fx += k * x21;
                fy += k * y21;
                fz += k * z21;
                count++;
            }
        }
    }
    fx /= count > 0 ? count : 1;
    fy /= count > 0 ? count : 1;
    fz /= count > 0 ? count : 1;
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("z", z1 + fz);
    FLAMEGPU->setVariable<float>("drift", cbrt(fx*fx + fy*fy + fz*fz));
    return flamegpu::ALIVE;
}
"""

# A Callback host function, to check the progress of the model / validate the model.
class step_validation(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()
        # Static variables?
        self.prevTotalDrift = 3.402823e+38 # @todo - static
        self.driftDropped = 0 # @todo - static
        self.driftIncreased = 0 # @todo - static

    def run(self, FLAMEGPU):
        # This value should decline? as the model moves towards a steady equilibrium state
        # Once an equilibrium state is reached, it is likely to oscillate between 2-4? values
        totalDrift = FLAMEGPU.agent("Circle").sumFloat("drift");
        if totalDrift <= self.prevTotalDrift:
            self.driftDropped += 1
        else:
            self.driftIncreased += 1
        self.prevTotalDrift = totalDrift;
        print("{:.2f} Drift correct".format(100 * self.driftDropped / float(self.driftDropped + self.driftIncreased)))


# Utility function to get the cuberoot of a number without requiring numpy
def cbrt(x):
    root = abs(x) ** (1/3)
    return root if x >= 0 else -root

# Define some constants
AGENT_COUNT = 16384
ENV_MAX = math.floor(cbrt(AGENT_COUNT))
RADIUS = 2.0
VISUALISE_COMMUNICATION_GRID = False

# Define a method which when called will define the model, Create the simulation object and execute it.
def main():
    # Define the FLAME GPU model
    model = pyflamegpu.ModelDescription("example_model")

    # Define environment properties
    env = model.Environment()
    env.newPropertyFloat("repulse", 0.05)

    # Define the location message list
    message = model.newMessageSpatial3D("location")
    # A message to hold the location of an agent.
    message.newVariableID("id")
    # X Y Z are implicit for spatial3D messages
    # Set Spatial3D message list parameters
    message.setRadius(RADIUS)
    message.setMin(0, 0, 0)
    message.setMax(ENV_MAX, ENV_MAX, ENV_MAX)

    # Define the Circle agent type including variables and messages
    agent = model.newAgent("Circle")
    agent.newVariableInt("id");
    agent.newVariableFloat("x");
    agent.newVariableFloat("y");
    agent.newVariableFloat("z");
    agent.newVariableFloat("drift");  # Store the distance moved here, for validation

    # Define agent functions
    output_message_description = agent.newRTCFunction("output_message", output_message)
    output_message_description.setMessageOutput("location")
    move_description = agent.newRTCFunction("move", move)
    move_description.setMessageInput("location")

    # Add a dependency that move requires outputMessage to have executed
    move_description.dependsOn(output_message_description)

    # Identify the root of execution
    model.addExecutionRoot(output_message_description)

    # Add the step function to the model.
    step_validation_fn = step_validation()
    model.addStepFunction(step_validation_fn)

    # Create the simulation object.
    simulation = pyflamegpu.CUDASimulation(model)
    
    # If visualisation is enabled, use it.
    if pyflamegpu.VISUALISATION:
        visualisation = simulation.getVisualisation()
        # Configure the visualiastion.
        INIT_CAM = ENV_MAX * 1.25
        visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM)
        visualisation.setCameraSpeed(0.01)
        vis_agent = visualisation.addAgent("Circle")
        # Position vars are named x, y, z so they are used by default
        # Set the model to use, and scale it.
        vis_agent.setModel(pyflamegpu.ICOSPHERE)
        vis_agent.setModelScale(1 / 10.0)
        # Optionally render the Subdivision of spatial messaging
        if VISUALISE_COMMUNICATION_GRID:
            ENV_MIN = 0
            DIM = int(math.ceil((ENV_MAX - ENV_MIN) / RADIUS))  # Spatial partitioning scales up to fit none exact environments
            DIM_MAX = DIM * RADIUS
            pen = visualisation.newLineSketch(1, 1, 1, 0.2)  # white
            # X lines
            for y in range(0, DIM + 1):
                for z in range(0, DIM + 1):
                    pen.addVertex(ENV_MIN, y * RADIUS, z * RADIUS)
                    pen.addVertex(DIM_MAX, y * RADIUS, z * RADIUS)
            # Y axis
            for x in range(0, DIM + 1):
                for z in range(0, DIM + 1):
                    pen.addVertex(x * RADIUS, ENV_MIN, z * RADIUS)
                    pen.addVertex(x * RADIUS, DIM_MAX, z * RADIUS)
            # Z axis
            for x in range(0, DIM + 1):
                for y in range(0, DIM + 1):
                    pen.addVertex(x * RADIUS, y * RADIUS, ENV_MIN)
                    pen.addVertex(x * RADIUS, y * RADIUS, DIM_MAX)
        # Activate the visualisation.
        visualisation.activate()

    # Initialise the simulation
    simulation.initialise(sys.argv)

    # Generate a population if an initial states file is not provided
    if not simulation.SimulationConfig().input_file:
        # Seed the host RNG using the cuda simulations' RNG
        random.seed(simulation.SimulationConfig().random_seed)
        # Generate a vector of agents
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        # Iterate the population, initialising per-agent values
        for i, instance in enumerate(population):
            instance.setVariableFloat("x", random.uniform(0, ENV_MAX))
            instance.setVariableFloat("y", random.uniform(0, ENV_MAX))
            instance.setVariableFloat("z", random.uniform(0, ENV_MAX))
        # Set the population for the simulation object
        simulation.setPopulationData(population)

    # Execute the simulation
    simulation.simulate()

    # If visualisation is enabled, end the visualisation
    if pyflamegpu.VISUALISATION:
        visualisation.join()

    # Ensure profiling / memcheck work correctly
    pyflamegpu.cleanup()

# If this python script is the entry point, execute the main method
if __name__ == "__main__":
    main()
