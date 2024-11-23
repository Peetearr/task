import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
import numpy as np

spring_coef = 9
damping_coef = 5
rest_angle = np.pi / 2



sys = chrono.ChSystemSMC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))
sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
body = [None]*4
motor = None

class Model():

    spring_coef = 9
    damping_coef = 5
    rest_angle = np.pi / 3
           
    def __init__(self):
        self.l_1 = 1
        self.l_2 = 1
        self.m = 10
        self.k = 20
        self.mat = chrono.ChContactMaterialSMC()
        self.bodies()
        self.joints()
        self.motors()
        # self.spin()

    def bodies(self):
        lenght = [self.l_1, self.l_2, 0, 0]
        pos = []
        for i in range(4):
            box = [0.01*(i<2) + 0.5*(i == 2) + 3*(i == 3),
                   lenght[i] + 0.5*(i == 2) + 1*(i == 3),
                   0.01*(i<2) + 0.5*(i == 2) + 3*(i == 3)]
            body[i] = chrono.ChBodyEasyBox(box[0],
                                           box[1],
                                           box[2], 
                                           1000)
            pos = pos + [((lenght[0]/2 + (lenght[0]/2 + lenght[1]/2)*(i>0)))*(i<=2) + (lenght[1]/2+0.25)*(i == 2) - 0.7*(i == 3)]
            body[i].SetMass(1*(i == 2)+0.2*(i != 2))
            body[i].SetPos(chrono.ChVector3d(0,
                                            pos[i],
                                            0))
            body_ct_shape = chrono.ChCollisionShapeBox(self.mat, 
                                                          box[0],
                                                          box[1],
                                                          box[2])
            body[i].AddCollisionShape(body_ct_shape)
            body[i].EnableCollision(i == 0 or i == 3)
            body[i].SetFixed(i == 3)
            sys.Add(body[i])
        lock = chrono.ChLinkLockPrismatic()
        lock.Initialize(body[3], body[2], chrono.ChFramed(chrono.ChVector3d(0, 2.25, 0), chrono.QuatFromAngleX(np.pi/2)))
        sys.AddLink(lock)

    def joints(self):
        rev = []
        lenght = [self.l_1, self.l_2]
        for i in range(2):
            rev = rev + [chrono.ChLinkLockRevolute()]
            rev[i].Initialize(body[i], body[i+1], chrono.ChFramed(chrono.ChVector3d(0, body[i].GetPos().y + 0.5*lenght[i], 0), chrono.ChQuaterniond(1, 0, 0, 0)))
            sys.AddLink(rev[i])
        
        spring = chrono.ChLinkRSDA()
        spring.SetSpringCoefficient(self.k)
        spring.SetDampingCoefficient(1)
        spring.Initialize(body[0],body[1],
                        chrono.ChFramed(chrono.ChVector3d(0, 2*body[0].GetPos().y, 0), chrono.QuatFromAngleZ(0*chrono.CH_PI)))
        spring.SetRestAngle(rest_angle)
        sys.AddLink(spring)
        spring.AddVisualShape(chrono.ChVisualShapeSpring(0.05, 200, 15))
    
    def motors(self):
        motor_1 = chrono.ChLinkMotorRotationTorque()
        motor_1.Initialize(body[1], body[2],
                        chrono.ChFramed(chrono.ChVector3d(0, body[2].GetPos().y - 0.25, 0)))
        sys.Add(motor_1)
        motor_2 = chrono.ChLinkMotorRotationTorque()
        motor_2.Initialize(body[0], body[1],
                        chrono.ChFramed(chrono.ChVector3d(0, 2 * body[0].GetPos().y, 0)))
        sys.Add(motor_2)
        
        self.torque_motor_1 = chrono.ChFunctionSetpoint()
        self.torque_motor_2 = chrono.ChFunctionSetpoint()
        motor_1.SetTorqueFunction(self.torque_motor_1)
        motor_2.SetTorqueFunction(self.torque_motor_2)

m = Model()
print(motor)

vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(sys)
vis.SetWindowSize(1024, 768)
vis.SetWindowTitle('Hopper')
vis.Initialize()
vis.AddSkyBox()
vis.AddCamera(chrono.ChVector3d(1, 1, 3), chrono.ChVector3d(0, 1, 0))
vis.AddTypicalLights()
t_0 = 0
error_0 = 0
angle = (body[1].GetRot()).GetCardanAnglesXYZ().z
print(angle)

while vis.Run():
    
    local_point = chrono.ChCoordsysd(chrono.ChVector3d(0,-0.5,0))
    error = -((body[0].GetCoordsys()*local_point).pos.x)
    angle = (body[1].GetRot()).GetCardanAnglesXYZ().z
    # print(angle)
    t = sys.GetChTime()
    Sp_1 = error*50 + 50*(error-error_0)/((t-t_0)*(t!=0)+ (t==0))
    Sp_2 = -(angle>0.9)*6.9469
    error_0 = error; t_0 = t
    m.torque_motor_1.SetSetpoint(Sp_1, t)
    m.torque_motor_2.SetSetpoint(Sp_2, t)
    
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    sys.DoStepDynamics(1e-3)
    # stop simulation after 2 seconds
    if sys.GetChTime() > 100:
        vis.GetDevice().closeDevice()
