#! /usr/bin/env python

import shlex, argparse, cmd
import random, time, os, sys, threading

################################################################

class Conversions(object):
    """Support conversions of storage units with SI-ish units."""
    
    import re as _re

    _spec = _re.compile('''^\s*(?P<digits>[0-9]+)(?P<frac>\.[0-9]*)?\s*(?P<spec>.*)\s*$''')

    # Not quite SI units...  We allow suffix of B for Bytes.
    #
    # We use lowercase for powers of 10, uppercase for powers of 1024,
    # even when there is ambiguity in SI units or no lowercase form.
    #
    _mult = {
        'k': 1000,
        'kB': 1000,
        'K': 1024,
        'KB': 1024,
        'KiB': 1024,
        'm': 1000 * 1000,
        'mB': 1000 * 1000,
        'M': 1024 * 1024,
        'MB': 1024 * 1024,
        'MiB': 1024 * 1024,
        'g': 1000 * 1000 * 1000,
        'gB': 1000 * 1000 * 1000,
        'G': 1024 * 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
        'GiB': 1024 * 1024 * 1024,
        't': 1000 * 1000 * 1000 * 1000,
        'tB': 1000 * 1000 * 1000 * 1000,
        'T': 1024 * 1024 * 1024 * 1024,
        'TB': 1024 * 1024 * 1024 * 1024,
        'TiB': 1024 * 1024 * 1024 * 1024,
        'p': 1000 * 1000 * 1000 * 1000 * 1000,
        'pB': 1000 * 1000 * 1000 * 1000 * 1000,
        'P': 1024 * 1024 * 1024 * 1024 * 1024,
        'PB': 1024 * 1024 * 1024 * 1024 * 1024,
        'PiB': 1024 * 1024 * 1024 * 1024 * 1024,
        'e': 1000 * 1000 * 1000 * 1000 * 1000 * 1000,
        'eB': 1000 * 1000 * 1000 * 1000 * 1000 * 1000,
        'E': 1024 * 1024 * 1024 * 1024 * 1024 * 1024,
        'EB': 1024 * 1024 * 1024 * 1024 * 1024 * 1024,
        'EiB': 1024 * 1024 * 1024 * 1024 * 1024 * 1024,
        }

    _div = (
        (_mult['KiB'],            1, 'B'),
        (_mult['MiB'], _mult['KiB'], 'KiB'),
        (_mult['GiB'], _mult['MiB'], 'MiB'),
        (_mult['TiB'], _mult['GiB'], 'GiB'),
        (_mult['PiB'], _mult['TiB'], 'TiB'),
        (_mult['EiB'], _mult['PiB'], 'PiB'),
        (        None, _mult['EiB'], 'EiB'))
        

    @classmethod
    def datasize2int(cls,s):
        """Convert a data size specification into a long integer."""
        m = cls._spec.match(s)
        if m is not None:
            digits = m.group('digits')
            frac = m.group('frac')
            if frac is not None:
                i = float(digits+frac)
            else:
                i = long(digits)
            ds = m.group('spec')
            if ds == '':
                return i
            else:
                try:
                    return long(i * cls._mult[ds])
                except:
                    pass
        raise DataSizeError, s

    @classmethod
    def int2datasize(cls,v):
        """Convert an integer into a data size specification string."""
        for b,d,ds in cls._div:
            if b is None or v < b:
                return str(v/d) + ds

class DataSizeError(Exception):
    def __init__(self, ds):
        self._ds = ds

    def __repr__(self):
        return repr(self._ds)

################################################################

class BlockDeviceOps(object):
    """Collect common operations on block devices."""
    # FIXME: Derive a class for Linux, have an abstract base class.

    @classmethod
    def set_io_variable(cls, dev, var, val):
        """Set a I/O elevator variable on the device."""
        import os.path, glob

        # Find the device under /sys/block.  We may need to resolve
        # symlinks.
        dev = os.path.realpath(dev)
        key = os.path.basename(dev)
        if not os.path.isdir('/sys/block/' + key):
            sys.stderr.write('Unable to manipulate I/O tunable %s for %s\n'
                             % (var, dev) )
            return
        # We need to manage slave settings first.
        for s in glob.iglob('/sys/block/'+key+'/slaves/*'):
            cls.set_io_variable(s, var, val)

        # Now set it for the master.
        with open('/sys/block/'+key+'/'+var, 'w') as f:
            f.write(str(val))

    @classmethod
    def set_io_scheduler(cls, dev, sched):
        """Set the I/O scheduler"""
        cls.set_io_variable(dev, 'queue/scheduler', sched)

    @classmethod
    def set_io_transfer_size(cls, dev, s):
        """Set the I/O transfer size to the device"""
        cls.set_io_variable(dev, 'queue/max_sectors_kb', s)

    @classmethod
    def set_io_readahead_size(cls, dev, s):
        """Set the I/O readahead size to the device"""
        cls.set_io_variable(dev, 'queue/read_ahead_kb', s)


################################################################


class TargetData(object):
    """Describe a target, typically a block device."""

    @staticmethod
    def build_parser(sp, cmdname):
        p = sp.add_parser(cmdname, description='Define a target for testing.')
        p.add_argument(
            'target', type=str,
            help='name of the target')
        p.add_argument(
            'device', type=str,
            help='path to device')
        p.add_argument(
            '--size', type=Conversions.datasize2int, required=True,
            help='size of the disk array')
        p.add_argument(
            '--count', type=int, default=1,
            help='number of data disks in disk array')
        # Stripe/segment size
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument(
            '--segment', type=Conversions.datasize2int,
            help='segment length on each disk of array')
        g.add_argument(
            '--stripe', type=Conversions.datasize2int,
            help='stripe size of the array')
        return p

    def __init__(self, n):
        self.target = n.target
        self.device = n.device
        self.dev_length = n.size
        self.data_disks = n.count
        if n.segment is not None:
            self.segment_len = n.segment
            self.stripe_len = n.count * n.segment
        else:
            self.stripe_len = n.stripe
            self.segment_len = n.stripe / n.count

################################################################

class TestData(object):
    """Describe a collection of test parameters"""

    @staticmethod
    def build_parser(sp, cmdname):
        p = sp.add_parser(cmdname, description='Define an i/o test')
        p.add_argument(
            'testname', type=str,
            help='name of the test')
        p.add_argument(
            '--blocksize', type=Conversions.datasize2int,
            default=Conversions.datasize2int('4MiB'),
            help='File system block size')
        p.add_argument(
            '--misalignment', type=Conversions.datasize2int,
            default=Conversions.datasize2int('0'),
            help='IOP misalignment factor from start of block')
        p.add_argument(
            '--iop', type=Conversions.datasize2int,
            default=Conversions.datasize2int('4MiB'),
            help='Size of each IOP')
        p.add_argument(
            '--transfer', type=Conversions.datasize2int,
            default=Conversions.datasize2int('2GiB'),
            help='Total amount of data written during test')
        p.add_argument(
            '--scheduler', type=str,
            choices=('noop', 'deadline', 'cfq', 'anticipatory'),
            default='noop',
            help='I/O elevator scheduler to use')
        p.add_argument(
            '--hwtransfer', type=Conversions.datasize2int,
            help='Hardware transfer size to use for each IOP')
        p.add_argument(
            '--hwreadahead', type=Conversions.datasize2int,
            help='Hardware readahead to use for each IOP')
        return p

    def __init__(self, n):
        self.testname = n.testname
        self.block_size = n.blocksize
        self.misalignment = n.misalignment
        self.iop_size = n.iop
        self.transfer_size = n.transfer
        self.scheduler = n.scheduler
        self.hwtransfer = n.hwtransfer
        self.hwreadahead = n.hwreadahead

################################################################

class HostData(object):
    """Describe a host and how to communicate with it."""

    @staticmethod
    def build_parser(sp, cmdname):
        p = sp.add_parser(cmdname,
                          description='Describe a host for targets.')
        p.add_argument(
            'hostname', type=str,
            help='name of the host')
        p.add_argument(
            '--workload', type=str, default=sys.argv[0],
            help='location of workload script on remote system')
        return p

    def __init__(self, n):
        self.hostname = n.hostname
        self.workload = n.workload

################################################################

class BaseTestThread(threading.Thread):
    """Base class for test threads."""

    def __init__(self, ti, rank):
        threading.Thread.__init__(self)
        self.daemon = True
        self.ti = ti
        self.rank = rank
        self.d = os.open(self.ti.device, os.O_RDWR)

    def wait_join(self):
        """Wait for the thread to stop, handling signals."""
        while self.is_alive():
            self.join(1)


class WriteTestThread(BaseTestThread):
    """Thread for testing writes."""

    def run(self):
        """Run the thread's portion of the test."""
        d = self.d
        iop_size = self.ti.iop_size
        loclist = self.ti.loclist
        bytes = self.ti.bytes
        outfile = self.ti.outfile
        fails = 0
        for i in range(self.rank, self.ti.iop_cnt, self.ti.wthreads):
            os.lseek(d, loclist[i], os.SEEK_SET)
            if os.write(d, bytes[i:i+iop_size]) < iop_size:
                outfile.write('Short write!!! IOP #%i\n' % (i,))
                fails += 1
                # We might want this to be configurable
                if fails > 3:
                    break
        os.fsync(d)
        os.close(d)

class ReadTestThread(BaseTestThread):
    """Thread for testing reads."""

    def run(self):
        """Run the thread's portion of the test."""
        d = self.d
        iop_size = self.ti.iop_size
        loclist = self.ti.loclist
        bytes = self.ti.bytes
        outfile = self.ti.outfile
        fails = 0
        for i in range(self.rank, self.ti.iop_cnt, self.ti.rthreads):
            os.lseek(d, loclist[i], os.SEEK_SET)
            junk = os.read(d, iop_size)
            if len(junk) < iop_size:
                outfile.write('Short read!!! IOP %i\n' % (i,))
                fails += 1
                if fails > 3:
                    break
                continue
            if junk != bytes[i:i+iop_size]:
                outfile.write('Read does not match! IOP # %i\n' % (i,))
                fails += 1
                if fails > 3:
                    break
        os.close(d)

class TestInstance(object):
    def __init__(self, test, target, outfile, wthreads, rthreads):
        """Build an instance that conducts the different phases of
        testing."""

        self.testname = test.testname
        self.targetname = target.target
        self.outfile = outfile
        self.wthreads = wthreads
        self.rthreads = rthreads
        self.device = target.device
        self.dev_length = target.dev_length
        self.segment_len = target.segment_len
        self.stripe_len = target.stripe_len
        self.transfer_size = test.transfer_size
        if test.block_size is not None:
            self.block_size = test.block_size
        else:
            self.block_size = target.segment_len * target.data_disks
        if test.iop_size is not None:
            self.iop_size = test.iop_size
        else:
            self.iop_size = self.block_size
        self.misalignment = test.misalignment
        self.scheduler = test.scheduler
        self.hwtransfer = test.hwtransfer
        self.hwreadahead = test.hwreadahead
        #
        self.iop_cnt = self.transfer_size / self.iop_size

    def generate_random_bytes(self):
        """Generate some random data for writing."""
        bytes = []
        for i in range(self.iop_size + self.iop_cnt):
            bytes.append( chr(random.randint(0,255)) )
        bytes = ''.join(bytes)
        self.bytes = bytes

    def generate_random_positions(self):
        """Generate a set of positions for random I/O testing"""
        dev_length = self.dev_length
        iop_size = self.iop_size
        iop_cnt = self.iop_cnt
        block_size = self.block_size
        misalignment = self.misalignment

        # What is the maximum misalignment?
        if misalignment > 0:
            mmf = (block_size / misalignment)
        else:
            mmf = 0
        max_misalignment =  mmf*misalignment

        #
        max_block_loc = ((dev_length - iop_size - max_misalignment)
                 / block_size) * block_size

        # Get a collection of random locations for writing
        loclist = random.sample(xrange(0,max_block_loc, block_size), iop_cnt)
        if misalignment > 0:
            for i in range(len(blocklist)):
                loclist[i] += random.randint(0,mmf)*misalignment

        self.loclist = loclist


    def prep_test(self):
        self.generate_random_bytes()
        self.generate_random_positions()
        dev = self.device
        BlockDeviceOps.set_io_scheduler(dev, self.scheduler)
        if self.hwtransfer is not None:
            BlockDeviceOps.set_io_transfer_size(dev, self.hwtransfer)
        if self.hwreadahead is not None:
            BlockDeviceOps.set_io_readahead_size(dev, self.hwreadahead)

    def run_test(self):

        # Get values
        outfile = self.outfile
        iop_size = self.iop_size
        iop_cnt = self.iop_cnt
        block_size = self.block_size
        transfer_size = self.transfer_size
        loclist = self.loclist
        bytes = self.bytes
        device = self.device

        # Print some information about the test.
        outfile.write('\n\nTest %s on target %s\n' 
                      % (self.testname, self.targetname))
        outfile.write('Device = %s\n' % (device,))
        outfile.write('iop_size = %s\n' % (Conversions.int2datasize(iop_size),))
        outfile.write('block_size = %s\n'
                      % (Conversions.int2datasize(block_size),))
        outfile.write('transfer_size = %s\n'
                      % (Conversions.int2datasize(transfer_size),))
        outfile.write('segment_len = %s\n'
                      % (Conversions.int2datasize(self.segment_len),))
        outfile.write('stripe_len = %s\n'
                      % (Conversions.int2datasize(self.stripe_len),))
        outfile.write('misalignment = %s\n'
                      % (Conversions.int2datasize(self.misalignment),))
        outfile.write('scheduler = %s\n' % (self.scheduler,))
        if self.hwtransfer is None:
            outfile.write('hwtransfer = \n')
        else:
            outfile.write('hwtransfer = %s\n'
                          % (Conversions.int2datasize(self.hwtransfer),))
        if self.hwreadahead is None:
            outfile.write('hwreadahead = \n')
        else:
            outfile.write('hwreadahead = %s\n'
                          % (Conversions.int2datasize(self.hwreadahead),))
        outfile.write('Minimum seek = %s\n' % (min(loclist),))
        outfile.write('Maximum seek = %s\n' % (max(loclist),))

        # Set up threads for writing.
        writers = [ WriteTestThread(self, i) for i in range(self.wthreads) ]

        # Perform writes.
        startw = time.time()
        for w in writers:
            w.start()
        for w in writers:
            w.wait_join()
        endw = time.time()

        # Print results for writing
        timeW = endw - startw
        outfile.write('write time = %g seconds  (%g MiB/sec)\n' % \
                      ( timeW,
                        transfer_size/timeW/1000000 ) )

        # Set up threads for reading.
        readers = [ ReadTestThread(self, i) for i in range(self.rthreads) ]

        # Perform reads.
        startr = time.time()
        for r in readers:
            r.start()
        for r in readers:
            r.wait_join()
        endr = time.time()

        # Print results for reading
        timeR = endr - startr
        outfile.write('read time = %g seconds  (%g MiB/sec)\n' % \
                      ( timeR,
                        transfer_size/timeR/1000000 ) )

################################################################

class RunTest(object):
    @staticmethod
    def build_parser(cmdname):
        p = argparse.ArgumentParser(
            prog=cmdname,
            description='Run I/O test.')
        p.add_argument('test', type=str, help='Name of a test')
        p.add_argument('target', type=str, help='Name of target')
        p.add_argument('--wthreads', type=int, default=1,
                       help='Number of threads for writing')
        p.add_argument('--rthreads', type=int, default=1,
                       help='Number of threads for reading')
        return p
    
    def __init__(self, n, targets, tests, outfile):
        try:
            test = tests[n.test]
        except:
            sys.stderr('Unknown test %s\n' % (n.test,))
            sys.exit(1)
        try:
            target = targets[n.target]
        except:
            sys.stderr('Unknown target %s\n' % (n.target,))
            sys.exit(1)

        # Create the test object.
        self.ti = TestInstance(test, target, outfile,
                               n.wthreads, n.rthreads)

    def run_test(self):
        """Run the test."""

        self.ti.prep_test()
        self.ti.run_test()
        

################################################################


class IOTester(cmd.Cmd):

    def __init__(self, cmdfile, outfile):
        # Create parsers
        ap = self.ap_parse_define = argparse.ArgumentParser(
            prog='define',
            description='Define an object for the I/O tester')
        sp = ap.add_subparsers()

        p = TargetData.build_parser(sp, 'target')
        p.set_defaults(func=self._do_define_target)
        p = TestData.build_parser(sp, 'test')
        p.set_defaults(func=self._do_define_test)
        p = HostData.build_parser(sp, 'host')
        p.set_defaults(func=self._do_define_host)
        self.ap_parse_run = RunTest.build_parser('run')
        self._cmd = ''
        cmd.Cmd.__init__(self, stdin=cmdfile)
        if cmdfile != sys.stdin:
            self.use_rawinput = False
            self.prompt = ''
        else:
            self.prompt = 'iot: '
        #
        self.targets = {}
        self.tests = {}
        self.hosts = {}
        self.outfile = outfile

    def emptyline(self):
        return False

    def precmd(self, line):
        # Help deal with continued lines
        if line == '' or line[-1] != '\\':
            # Command is finished here
            res = self._cmd + line
            self._cmd = ''
            lres = res.lstrip()
            if lres != '' and lres[0] == '#':
                return ''
            return res
        if line != '':
            # Continued command (ends with backslash)
            self._cmd = self._cmd + line[:-1]
            return ''

    def postcmd(self, stop, line):
        # Handle prompting for continued lines
        if self.use_rawinput:
            if self._cmd == '':
                self.prompt = 'iot: '
            else:
                self.prompt = '____ '
        return stop

    # "exit"

    def do_exit(self, cs):
        return True
    do_EOF = do_exit

    def help_exit(self):
        print 'usage: exit'
        print ''
        print 'Exit from iot'
    help_EOF = help_exit

    # "define"

    def do_define(self, cs):
        L = shlex.split(cs)
        try:
            n = self.ap_parse_define.parse_args(L)
        except SystemExit as e:
            return False
        except:
            return True
        n.func(n)
        return False

    def help_define(self):
        self.ap_parse_define.print_help()

    def _do_define_target(self, n):
        self.targets[n.target] = TargetData(n)

    def _do_define_test(self, n):
        self.tests[n.testname] = TestData(n)

    def _do_define_host(self, n):
        self.hosts[n.hostname] = HostData(n)

    # "run"

    def do_run(self, cs):
        L = shlex.split(cs)
        try:
            n = self.ap_parse_run.parse_args(L)
        except SystemExit as e:
            return False
        except:
            return True
        r = RunTest(n, self.targets, self.tests, self.outfile)
        r.run_test()
        return False

    def help_run(self):
        self.ap_parse_run.print_help()


def main():

    # Define command line parser.
    p = argparse.ArgumentParser(description='Test I/O device performance.')
    p.add_argument('cmdfile', type=argparse.FileType('r'),
                   nargs='?', default='-')
    p.add_argument('--output', type=argparse.FileType('w'), default='-',
                   help='File for output of test results')
    n = p.parse_args()
    t = IOTester(n.cmdfile, n.output)
    t.cmdloop()
    sys.exit(0)

if __name__ == '__main__':
    main()
