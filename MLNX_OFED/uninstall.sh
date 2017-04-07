#! /usr/bin/perl
#
# Copyright (c) 2013 Mellanox Technologies. All rights reserved.
#
# This Software is licensed under one of the following licenses:
#
# 1) under the terms of the "Common Public License 1.0" a copy of which is
#    available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/cpl.php.
#
# 2) under the terms of the "The BSD License" a copy of which is
#    available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/bsd-license.php.
#
# 3) under the terms of the "GNU General Public License (GPL) Version 2" a
#    copy of which is available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/gpl-license.php.
#
# Licensee has the right to choose one of the above licenses.
#
# Redistributions of source code must retain the above copyright
# notice and one of the license notices.
#
# Redistributions in binary form must reproduce both the above copyright
# notice, one of the license notices in the documentation
# and/or other materials provided with the distribution.

use strict;
use File::Basename;
use File::Path;
use File::Find;
use File::Copy;
use Cwd;
use Term::ANSIColor qw(:constants);

my $PREREQUISIT = "172";
my $MST_START_FAIL = "173";
my $NO_HARDWARE = "171";
my $SUCCESS = "0";
my $DEVICE_INI_MISSING = "2";
my $ERROR = "1";
my $EINVAL = "22";
my $ENOSPC = "28";
my $NONOFEDRPMS = "174";

$ENV{"LANG"} = "C";

if ($<) {
	print RED "Only root can run $0", RESET "\n";
	exit $PREREQUISIT;
}

$| = 1;

my $PACKAGENAME = `ofed_info -s | sed -e 's/://'`;
chomp $PACKAGENAME;

my $PACKAGE     = 'OFED';
my $ofedlogs = "/tmp/$PACKAGE.$$.logs";
mkpath([$ofedlogs]);

my $dry_run = 0;
my $quiet = 0;
my $force = 0;
my $verbose = 0;
my $log;
my $unload_modules = 0;

my $WDIR    = dirname($0);
chdir $WDIR;
my $CWD     = getcwd;

my $builddir = "/var/tmp/";

my %selected_for_uninstall;
my @dependant_packages_to_uninstall = ();
my %non_ofed_for_uninstall = ();
my %remove_debs_hs = ();
my $keep_mft = 0;
my $is_mlnx_en = 0;
my $mlnx_en_pkgs = "";
my $mlnx_en_only_pkgs = "mlnx.*en|mstflint|ofed-scripts|mlnx-fw-updater|^rdma\$";
my $mlnx_en_rdma_pkgs = "$mlnx_en_only_pkgs|mlnx-ofed-kernel|ibverbs|libmlx4|librdmacm|libvma|sockperf";

if ($PACKAGENAME =~ /mlnx-en/i) {
	$is_mlnx_en = 1;
	$keep_mft = 1;
}

my @remove_debs = qw( ofed-docs mlnx-ofed-kernel-utils mlnx-ofed-kernel-dkms mlnx-ofed-kernel-modules infiniband-diags ibutils rds-tools srptools libsdp1 libsdp-dev sdpnetstat perftest dapl-dev dapl1 dapl2-utils libdapl-dev libdapl2 dapl1-utils compat-dapl-dev compat-dapl1 libopensm-dev libopensm-devel libopensm opensm-doc opensm opensm-libs libopensm2-dev libopensm2 librdmacm1-dbg librdmacm-dev rdmacm-utils librdmacm1 libipathverbs1-dbg libipathverbs-dev libipathverbs1 ibacm-dev ibacm ibsim-utils ibsim libibmad libibmad-dev libibmad1 libibmad-devel libibmad-static libibumad libibumad-dev libibumad-devel libibumad-static libibumad1 libibcm-dev libibcm1 libibdm-dev libibdm1 libumad2sim0 libmlx4 libmlx4-1 libmlx4-1-dbg libmlx4-dev libmlx5 libmlx5-1 libmlx5-1-dbg libmlx5-dev librxe-1 librxe-dev librxe-1-dbg ibverbs-utils libibverbs1-dbg libibverbs-dev libibverbs1 ar-mgr cc-mgr dump-pr ofed-scripts mft kernel-mft-dkms mft-compat mft-devel mft-devmon mft-devmondb mft-int mft-tests mstflint ibdump mxm ucx fca openmpi openshmem mpitests knem knem-dkms ummunotify ummunotify-dkms libvma libvma-utils libvma-dev libvma-dbg sockperf srptools iser-dkms srp-dkms libmthca-dev libmthca1 libmthca1-dbg ibdump mlnx-ethtool mlnx-fw-updater knem-modules iser-modules isert-modules srp-modules ummunotify-modules kernel-mft-modules libosmvendor libosmvendor4 libosmcomp libosmcomp3 mlnx-en mlnx-en-utils mlnx-en-dkms mlnx-en-modules mlnx-sdp-dkms mlnx-sdp-modules mlnx-rds-dkms mlnx-rds-modules mlnx-nfsrdma-dkms mlnx-nfsrdma-modules mlnx-nvme-rdma-dkms mlnx-nvme-rdma-modules mlnx-nvmet-rdma-dkms mlnx-nvmet-rdma-modules mlnx-rdma-rxe-dkms mlnx-rdma-rxe-modules);

sub print_red
{
	print RED @_, RESET "\n";
}

sub print_green
{
	print GREEN @_, RESET "\n";
}

sub print_and_log
{
	my $msg = shift @_;
	print LOG $msg . "\n";
	print $msg . "\n" if ($verbose);
}

sub getch
{
	my $c;
	system("stty -echo raw");
	$c=getc(STDIN);
	system("stty echo -raw");
	# Exit on Ctrl+c or Esc
	if ($c eq "\cC" or $c eq "\e") {
		print "\n";
		exit 1;
	}
	print "$c\n";
	return $c;
}

sub is_installed_deb
{
	my $name = shift @_;

	my $installed_deb = `dpkg-query -l $name 2> /dev/null | awk '/^[rhi][iU]/{print \$2}'`;
	return ($installed_deb) ? 1 : 0;
}

sub get_all_matching_installed_debs
{
	my $name = shift @_;

	my $installed_debs = `dpkg-query -l "*$name*" 2> /dev/null | awk '/^[rhi][iU]/{print \$2}'`;
	return (split "\n", $installed_debs);
}

sub mark_for_uninstall
{
	my $package = shift @_;

	return if ($keep_mft and $package =~ /mft/);
	return if ($package =~ /^xen|ovsvf-config/);
	if (not $selected_for_uninstall{$package}) {
		if (is_installed_deb $package) {
			print_and_log "$package will be removed.";
			push (@dependant_packages_to_uninstall, "$package");
			$selected_for_uninstall{$package} = 1;
			if ( not exists $remove_debs_hs{$_} and `ofed_info 2>/dev/null | grep -i $package 2>/dev/null` eq "" and $package !~ /ofed-scripts|mlnx-ofed-/) {
				$non_ofed_for_uninstall{$package} = 1;
			}
		}
	}
}

sub get_requires
{
	my $package = shift @_;

	chomp $package;

	if ($package eq "rdma") {
		# don't remove packages that needs rdma package
		return;
	}

	my @what_requires = `/usr/bin/dpkg --purge --dry-run $package 2>&1 | grep "depends on" 2> /dev/null`;

	for my $pack_req (@what_requires) {
		chomp $pack_req;
		$pack_req =~ s/\s*(.+) depends.*/$1/g;
		print_and_log "get_requires: $package is required by $pack_req\n";
		get_requires($pack_req);
		mark_for_uninstall($pack_req);
	}
}

sub ex
{
	my $cmd = shift @_;
	my $sig;
	my $res;

	print_and_log "Running: $cmd";
	system("$cmd >> $log 2>&1");
	$res = $? >> 8;
	$sig = $? & 127;
	if ($sig or $res) {
		print_red "Failed command: $cmd";
		exit 1;
	}
}

sub usage
{
	print GREEN;
	print "\n Usage: $0 [options]\n";
	print "\n           --unload-modules     Run /etc/init.d/openibd stop before uninstall";
	print "\n           --force              Force uninstallation and remove packages that depends on MLNX_OFED";
	print "\n           --keep-mft           Don't remove MFT package";
	print "\n           -v|--verbose         Increase verbosity level";
	print "\n           --dry-run            Print the list of packages to be uninstalled without actually uninstalling them";
	print "\n           -q                   Set quiet - no messages will be printed";
	print RESET "\n\n";
}

sub is_configured_deb
{
	my $name = shift @_;

	my $installed_deb = `/usr/bin/dpkg-query -l $name 2> /dev/null | awk '/^rc/{print \$2}'`;
	return ($installed_deb) ? 1 : 0;
}

########
# MAIN #
########
sub main
{
	$log = "/tmp/ofed.uninstall.log";
	rmtree $log;
	open (LOG, ">$log") or die "Can't open $log: $!\n";

	for my $name (@remove_debs) {
		$remove_debs_hs{$name} = 1;
	}

	# parse arguments
	while ( $#ARGV >= 0 ) {
		my $cmd_flag = shift(@ARGV);

		if ($cmd_flag eq "--force") {
			$force = 1;
		} elsif ($cmd_flag eq "--unload-modules") {
			$unload_modules = 1;;
		} elsif ($cmd_flag eq "-v" or $cmd_flag eq "--verbose") {
			$verbose = 1;
		} elsif ($cmd_flag eq "-q" or $cmd_flag eq "--quiet") {
			$quiet = 1;
		} elsif ($cmd_flag eq "--dry-run") {
			$dry_run = 1;
		} elsif ($cmd_flag eq "--keep-mft") {
			$keep_mft = 1;
		} elsif ($cmd_flag eq "-h" or $cmd_flag eq "--help") {
			usage();
			exit 0;
		} else {
			usage();
			exit 0;
		}
	}

	warn("Log: $log\n") unless($quiet);

	if (is_installed_deb("mlnx-ofed-kernel-utils")) {
		$mlnx_en_pkgs = $mlnx_en_rdma_pkgs;
	} else {
		$mlnx_en_pkgs = $mlnx_en_only_pkgs;
	}

	if (not $force and not $dry_run) {
		print "\nThis program will uninstall all $PACKAGENAME packages on your machine.\n\n";
		print "Do you want to continue?[y/N]:";
		my $ans = getch();
		print "\n";
		if ($ans !~ m/[yY]/) {
			print "Uninstall was aborted.\n";
			exit $ERROR;
		}
	}

	print "Removing MLNX_OFED packages\n" unless($quiet);
	if ($dry_run) {
                print "Running in dry run mode. Packages will not be removed.\n";
                print_and_log "Running in dry run mode. Packages will not be removed.";
        }

        if ($unload_modules) {
            print "Unloading kernel modules...\n" if (not $quiet);

            if (not $dry_run) {
                system("/etc/init.d/openibd stop >> $ofedlogs/openibd_stop.log 2>&1");
                my $res = $? >> 8;
                my $sig = $? & 127;
                if ($sig or $res) {
                    print RED "Failed to unload kernel modules", RESET "\n";
                    exit $ERROR;
                }
            }
        }

	my @list_to_remove;
	foreach (@remove_debs){
		next if ($keep_mft and $_ =~ /mft/);
		next if ($is_mlnx_en and $_ !~ /$mlnx_en_pkgs/);
		next if ($_ =~ /^xen|ovsvf-config/);
		foreach (get_all_matching_installed_debs($_)) {
			if (not $selected_for_uninstall{$_}) {
				print_and_log "$_ will be removed.";
				push (@list_to_remove, $_);
				$selected_for_uninstall{$_} = 1;
				if ( not exists $remove_debs_hs{$_} and `ofed_info 2>/dev/null | grep -i $_ 2>/dev/null` eq "" and $_ !~ /ofed-scripts|mlnx-ofed-/) {
					$non_ofed_for_uninstall{$_} = 1;
				}
				get_requires($_);
			}
		}
	}

	if (not $force and keys %non_ofed_for_uninstall) {
		print "\nError: One or more packages depends on MLNX_OFED.\nThose packages should be removed before uninstalling MLNX_OFED:\n\n";
		print join(" ", (keys %non_ofed_for_uninstall)) . "\n\n";
		print "To force uninstallation use '--force' flag.\n";
		exit $NONOFEDRPMS;
	}

	print "The following packages will be removed:\n" unless($quiet);
	print join(" ", @list_to_remove) . " " . join(" ", @dependant_packages_to_uninstall) unless($quiet);
	print "\n";

	if (not $dry_run) {
		# verify that dpkg DB is ok
		print_and_log "Running: dpkg --configure -a";
		system("dpkg --configure -a >> $log 2>&1");
		print_and_log "Running: apt-get install -f";
		system("apt-get install -f  >> $log 2>&1");

		ex "apt-get remove -y @list_to_remove @dependant_packages_to_uninstall" if (scalar(@list_to_remove) or scalar(@dependant_packages_to_uninstall));
		foreach (@list_to_remove, @dependant_packages_to_uninstall){
			if (is_configured_deb($_)) {
				if (not /^opensm/) {
					ex "apt-get remove --purge -y $_";
				} else {
					system("apt-get remove --purge -y $_");
				}
			}
		}
		system("/sbin/modprobe -r knem > /dev/null 2>&1");
		system("sed -i '/knem/d' /etc/modules 2>/dev/null");
	}

	close(LOG);
}

main();

exit $SUCCESS;

