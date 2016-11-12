import getopt
import os
import socket
import sys
import time

PI = 3.14159265359

data_size = 2 ** 17

# Initialize help messages
ophelp = 'Options:\n'
ophelp += ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp += ' --port, -p <port>    TORCS port. [3001]\n'
ophelp += ' --id, -i <id>        ID for server. [SCR]\n'
ophelp += ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp += ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp += ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp += ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp += ' --debug, -d          Output full telemetry.\n'
ophelp += ' --help, -h           Show this help.\n'
ophelp += ' --version, -v        Show current version.'
usage = 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage = usage + ophelp
version = "2016-11-12"


class Client:
    def __init__(self, host=None, port=None, sid=None, track=None, track_type=None, gui=True):
        self.gui = gui
        self.host = 'localhost'
        self.port = 3001
        self.sid = 'SCR'
        self.track = 'forza'
        self.track_type = 'road'
        self.parse_the_command_line()

        if host:
            self.host = host
        if port:
            self.port = port
        if sid:
            self.sid = sid
        if track:
            self.track = track
        if track_type:
            self.track_type = track_type

        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        # == Initialize Connection To Server ==
        self.socket.settimeout(1)

        n_fail = 5
        while True:
            # This string establishes track sensor angles! You can customize them.
            # a= "-90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90"
            # xed- Going to try something a bit more aggressive...
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg = '%s(init %s)' % (self.sid, a)

            try:
                self.socket.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.socket.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(0.5)
                    if self.gui is False:
                        os.system(
                            'optirun torcs -nofuel -nodamage -nolaptime -r ~/.torcs/config/raceman/quickrace.xml &')
                    else:
                        os.system('optirun torcs -nofuel -nodamage -nolaptime &')
                        time.sleep(2)
                        os.system('sh autostart.sh')
                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                                         ['host=', 'port=', 'id=', 'steps=',
                                          'episodes=', 'track=', 'stage=',
                                          'debug', 'help', 'version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug = True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host = opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid = opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname = opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage = int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port = int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes = int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps = int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        '''Server's input is stored in a ServerState object'''
        if not self.socket: return
        sockdata = str()

        while True:
            try:
                # Receive server data
                sockdata, addr = self.socket.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
                # print "Waiting for data on %d.............." % self.port
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. " +
                        "You were in %d place.") %
                       (self.port, self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                # What do I do here?
                print("Server has restarted the race on %d." % self.port)
                # I haven't actually caught the server doing this.
                self.shutdown()
                return
            elif not sockdata:  # Empty?
                continue  # Try again.
            else:
                self.S.parse_server_str(sockdata)
                break  # Can now return from this function.

    def respond_to_server(self):
        if not self.socket: return
        try:
            message = repr(self.R)
            self.socket.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)
        # Or use this for plain output:
        # if self.debug: print self.R

    def shutdown(self):
        if not self.socket: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps, self.port)))
        self.socket.close()
        self.socket = None
        # sys.exit() # No need for this really.

    def setup_quick_race(self):
        print()


class ServerState:
    def __init__(self):
        self.servstr = str()
        self.actions = dict()

    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        servstrlist = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in servstrlist:
            w = i.split(' ')
            self.actions[w[0]] = destringify(w[1:])

    def __repr__(self):
        out= str()
        for k in sorted(self.d):
            strout= str(self.d[k])
            if type(self.d[k]) is list:
                strlist= [str(i) for i in self.d[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

class DriverAction:
    def __init__(self):
        self.actionstr = str()
        self.actions = {'accel': 0.2,
                        'brake': 0,
                        'clutch': 0,
                        'gear': 1,
                        'steer': 0,
                        'focus': [-90, -45, 0, 45, 90],
                        'meta': 0
                        }

    def limit_actions(self):
        self.actions['steer'] = limit_action(self.actions['steer'], -1, 1)
        self.actions['brake'] = limit_action(self.actions['brake'], 0, 1)
        self.actions['accel'] = limit_action(self.actions['accel'], 0, 1)
        self.actions['clutch'] = limit_action(self.actions['clutch'], 0, 1)
        if self.actions['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.actions['gear'] = 0
        if self.actions['meta'] not in [0, 1]:
            self.actions['meta'] = 0
        if type(self.actions['focus']) is not list or min(self.actions['focus']) < -180 or max(
                self.actions['focus']) > 180:
            self.actions['focus'] = 0

    def __repr__(self):
        self.limit_actions()
        out = str()
        for k in self.actions:
            out += '(' + k + ' '
            v = self.actions[k]
            if not type(v) is list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out
        return out + '\n'

def drive_example(client):
    S, R = client.S.actions, client.R.actions
    target_speed = 100

    # Steer To Corner
    R['steer'] = S['angle'] * 10 / PI
    # Steer To Center
    R['steer'] -= S['trackPos'] * .10

    # Throttle Control
    if S['speedX'] < target_speed - (R['steer'] * 50):
        R['accel'] += .01
    else:
        R['accel'] -= .01
    if S['speedX'] < 10:
        R['accel'] += 1 / (S['speedX'] + .1)

    # Traction Control System
    if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) -
            (S['wheelSpinVel'][0] + S['wheelSpinVel'][1]) > 5):
        R['accel'] -= .2

    # Automatic Transmission
    R['gear'] = 1
    if S['speedX'] > 50:
        R['gear'] = 2
    if S['speedX'] > 80:
        R['gear'] = 3
    if S['speedX'] > 110:
        R['gear'] = 4
    if S['speedX'] > 140:
        R['gear'] = 5
    if S['speedX'] > 170:
        R['gear'] = 6
    return


def limit_action(v, lo, hi):
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return v


def destringify(s):
    if not s:
        return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]


# ================ MAIN ================
if __name__ == "__main__":
    client = Client(port=3001)
    maxSteps = 100000  # 50steps/second

    for step in range(maxSteps, 0, -1):
        client.get_servers_input()
        drive_example(client)
        client.respond_to_server()
    client.shutdown()
