#
# Copyright (c) 2012 Mellanox Technologies. All rights reserved.
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
#
#
#  $Id: ofed-docs.spec 7948 2006-06-13 12:42:34Z vlad $
#

Summary: OFED docs
Name: ofed-docs
Version: @VERSION@
Release: 0
License: GPL/BSD
Url: http://mellanox.com/content/pages.php?pg=products_dyn&product_family=26&menu_section=34#tab-three
Group: Documentation/Man
Source: %{name}-%{version}.tar.gz
BuildRoot: %{?build_root:%{build_root}}%{!?build_root:/var/tmp/%{name}-%{version}-root}
Vendor: OpenFabrics
%description
OpenFabrics documentation

%prep
%setup -q -n %{name}-%{version}

%install
mkdir -p $RPM_BUILD_ROOT%{_defaultdocdir}/%{name}-%{version}
cp -a * $RPM_BUILD_ROOT%{_defaultdocdir}/%{name}-%{version}

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
%{_defaultdocdir}/%{name}-%{version}

%changelog
* Sun Mar 25 2007 Vladimir Sokolovsky <vlad@mellanox.co.il>
- Changed prefix
* Thu Jul 27 2006 Vladimir Sokolovsky <vlad@mellanox.co.il>
- Changed version to 1.1
* Tue Jun  6 2006 Vladimir Sokolovsky <vlad@mellanox.co.il>
- Initial packaging
