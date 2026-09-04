#!/usr/bin/env python3
"""Build a clean, portable copy of the polished Colab notebook.

The builder first looks for the maintained notebook, then for another valid
non-archival notebook. If none is available, it reconstructs the same 19-cell
source notebook from an embedded, compressed fallback template. By default it
never overwrites either its source or the submitted notebook.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAME = "Cosmic_Ray_ML_Colab.ipynb"
DEFAULT_SOURCE = ROOT / NOTEBOOK_NAME
DEFAULT_OUTPUT = ROOT / "build" / NOTEBOOK_NAME
DEFAULT_REPO_SLUG = "smaiti7/cosmic-ray-ml_sm"
EXCLUDED_SEARCH_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    "archive",
    "build",
    "full_runs",
}
REQUIRED_SECTIONS = (
    "# Cosmic-Ray Shower Detection and Reconstruction",
    "## 1. Physics objective and model choice",
    "## 2. Configuration",
    "## 3. Environment and GPU check",
    "## 4. Download and prepare the dataset",
    "## 5. Inspect geometry and one normalized waveform",
    "## 6. Architecture check",
    "## 7. Training and validation-controlled model selection",
    "## 8. Evaluation",
    "## 9. Full-run model comparison",
    "## 10. Interpretation and conclusion",
)

# Source-only snapshot of the current polished notebook. Compression keeps this
# independent fallback compact; its SHA-256 guards against accidental damage.
EMBEDDED_TEMPLATE_SHA256 = "e529eac44b7f8fdf8ac5e19f3adb08f86ca4e0844324da7e1326d68b8ccb2cb4"
EMBEDDED_TEMPLATE_B85 = (
    b"c-qZ8?NS>_lCM%L$HqvE)JPZ%$j3x57|(duHXO!tu>nN8)GA56wA9+J7NEsM+^>6pi;K7ix;NP;xy-EZRtp%fBkmkCKzCJTRpobP"
    b"W%Yd{5m_c1{fm40D=wFF(daj(ytp3bx3fmOkq+UxH+r?Z*WG>z&r?zIAuoBOe_v^PXg6fOC=vj0(Rjg*ayd<t&KX~_Pm}ys6zq*C"
    b"MN+2ujPco!orxr$$+B3`t4o}Veq8+F;$$vn?A;7NWPEVdoRsBU_BS^Ze2fYq1uv3GG|uyJCZZ&tZgMF_DOHEgfDgwak~iblbB2!7"
    b"a<Uj~$|+CFbQe0(stdmCOtV-{HwRfh*o4N}rrJ;p8;U=?k7;dDI$zEPEw}iMjnlj+`5+T)mX~6X=huDMx0tg}us}B8Qe^2&uz9wS"
    b"j5EMMS>#zJhU~B`#f*^Sx{Et_C*j@3hMVGil1d94Al!&TvT`C=A^x^V3jj4Q#JtE8A!Rxnx7kf97d%V<&JlW>Ap%k)#G(*wRu-@|"
    b">Xy&TbZUEv8=fu92%y$9ABqfSo6dQW%6t~FcO{!}nE5t`b&6y`YsYS~?-tu(Utm$m^_M>ER!#snYzSOXOkq-4rU^Vvi3?~dHp&Y{"
    b"^w(zeXpv<w825*hXE4~V88(`iIgpkuBy1Z8kJ1^>I)B6cmP^2hTx6w;+<J~QjG>x_A^1+Qn3=8!@+KV$nh>eJvhmjpSSU+@onfcf"
    b"Wj^oxa{Ad0=LOQCgn=*xV23C%TqNlr)hil_Ld<|R5&Ks}Ob6iy!}D3T4ACN^#??dnaS7e$`-5FR+O3mN+ZIab3)YL+>0~L>M6!JF"
    b"SCq{g;Zu8(r-^XLJLbs*7z5ap7#B7v*~Z3JZ<lDAlz{(Rej|VmQ@OFxX15?(&|!Oz0cQy|!2WYWoa()?5wTC{7$~NoN&^<wV0a%F"
    b"lorT3pg{oSBn8nL@Z@@2<O^77W#BANC2%zmD&t6H;B*PBU9gl?%@jJrY=cE11GYFvYSW%Uc7|zE`aBF+Qqv=I47j4CV$iLKW%(Ew"
    b"qnv<hFx|vV6yqiFqx83@#=JD(I!O-Xp64<p-v83PYqMqR%j&qwLah>kej6JnMLI^NcWPr}L+^#p=fJ2+IY<r^=)GY_A3w4(N5I8$"
    b"m24pE!=ZHVlRPhm0KF7asR2G77h;Uhj4O6AeI*DX%H)*GYe4)|aL^pGgh9L*La&?D*CzZfoi3(PMXQZK73Q~U!4nWMR8>n~xe_1!"
    b"<3Ik>tq>mwOTLQ4DF9qL8w#)_LuA-|<i^0BL;-89;1v~Mj+oros8Plit02-D2LT3Mb0-`|uro1y2KoV|-yv6|;|Wla!6Z_cM_gT+"
    b"046oqbUH`N0ji;m+MUfwFkv+bUr17C&(H!wbbCF{x9hZkZL4YlVB({EHcH0}rEVNn-oolR>=UGgfs$;qYrsaR-OoOMj6a;bIc5j!"
    b"vO%i;vhl?<V#E9Rx5Kk{hab<WN3OZmbfO=h0V_(;lb8xVL~?1vq4eR1n6aY&YY>tEE&-x7NO1yMqHOeM0A1tgmDu0iPxz}k9N9L&"
    b"QNAeWixPdKD<5HsPAr$Sbp0fk;p1YU4KTeO75S9Sc{$0_fu=J%g-3@04~w~s#Ox+5z{HM4*}QBVoqYQ6?kF~0E*ou!=@wJ}Ubt7N"
    b"6)qcy%}WKlN`!S2P(KCTQK?0s-Lrzujd<uOK7Trne>pk%<^6H|=Iq^XxQ(|wlge})|91HP^Wph%e181t+_d15L8{n!u@H{JDEDN+"
    b"8E7~V9yIE(5hWo?TF;J8PvXyK?}?j?cc*+fa_83haS`;<t0fRWpB;WY`jw!dnb%E9dw??u6tXk{vvP0I+qTGk@H>Y{RwEqN+9A%="
    b"$aLhUIPj!L$rn<!7ISW$?>@#yC+`n`rcGFn7Q7lD6RZGuG6KXcG9ld<nA8k&^T5l?#$Qf8pT$7PvyX=#juixhYu6M1f$#zSegpSV"
    b"92Z4i00*5ZmB<1ZoAiV*KD|3XIr|G6potM|G0$>7L{ov}HjZoryyljP^PHZY{OS1UJU%-)!Of$<z*L)P=>d@;fma5@mQaZ{10b8Q"
    b"O~nVTNZf({DVwcuZ!g$eqwM-D$?}=do;6wNyjTXkotQ^3pm_mQK;TGi8ms>7bifLMu+u}&yc-mt>glD&j%|x_G@V7T`qot&95Rty"
    b"AE3;%>_P<aM0FEqcD;;e1?bZyQ~T`$_J?S5(vp!@`Xw!YT@0SPtIiSrURfD7%}FNI-bSMiiu7fpX36v;TyqVfqvUqjY+<~XXE&nx"
    b"T;eL8h>0K8P*`#A85jj{B`p6!08j>D6(J`iJuC9O4B52u!xJv@a;j?4t%Sh)B5TaTumGRRcB3Z{Vpy%u`l?gI@c=<zCn(f=^TYBD"
    b"{njV~v5B475RBuuho9e{$0wi9Pd}f>Z{D2&EDtc^SDQr84x4Or*|-n1-a~wE^x4B@V+HQv+0n1cG*p1<74ZVsgw_~B`Dix(`=#NP"
    b"TAdYa`6CM3qZ_Fr#qOl_Pi<ud*RcG}^&4u{aQ`RJjfcO4)LUnw{l2x5*lK+%f`VmyKU%MVMSKv<CkanKhGN7aGN$G=_(mG~l85MB"
    b"H!s0_b9jE}v&A}Wn|=n<{#p#CSD9=bG}qI>Tcv}6YG$HT4a8UnE0~21djwXRi>R!&Z@2giJZq5g>EZdWPi?~QTSbdrtl5KELNvV|"
    b"rbTnk3k+q+610^yigVvEI4nGOguj?b^?vS9*k3Zio`LjG6S__4t49!ufM?h2)`^*8&^ZK>WfKJC1Nf+Y=EUN)@>=0lC^@fH<_X>#"
    b"j(sqS!NNK%a?Hj+AAj1_es6%7(8-{BwZjPsF?4a*UG0Q9F|f3lLen(dOgPwW80MoG2R<Ev4t9rwo$k(lU36vJsuAeRh#f0-!vqc4"
    b"PE4wi@8J2nIw9QF(E-Hi+Ky8e;Ap~UV}Y(4y~6d;dw>M7yLLXIb!J>on?|G@__qa<TYF+}_^MkQaND*=g2_75R&~ITBh?x(uS_o1"
    b"RJmI7bguruu$fy2>~H#ap?>SUFrrp=R3f~*E7yBx+7<b$)Tsdd3@(0&IU@hvY%!fLG0!=h`>(-WO1$V+c(gr$$jEFIHn1!w7!ma?"
    b"^hmvmF{=$IZ?@WDcY+N4>5nKj{o>f~mn?=n<~Kae5M!=8zK>GrnnLh2O!JQ$q-UVPe(&*oQGpDqGR8z#BH|gJie{G#)7q((ss{Nx"
    b"{S1U6$Pb=mBL1TxAXqj%1hy%wH7gIcv=*v5ooEQwPU)De)(`%eliaDyl^8}4qLZK*G5e`j$bO<hICp_ZIx`p%Jxf=-+!~B_2EAJ0"
    b"v29gdu^q8D00U!4niCH@Fl~+!t&LkHtv4E_Ny=3YU+X#D5e+9FfNZi}w8wr?2o!g7_$4t*TI7rIBwMomXct;3fyahsa+wtAyi_8j"
    b"QkLS51F~SeN?i<=B_3J;sv>qut5jJ+aR-_wDJ7eVDZ~|>DWA{j#E2}kY_lo97LuVc56?=Z5W~*)7E|W}qZH=Y?QZsV;6EnkSz4;("
    b"D(-WE8CsR(oq<%@;~&h(mI|5*af}Wm2MeRNN>bEhhOy^N(qDkIFPNWh-nH2O{5K}Rn?3d;`+E1a^@TF@$;9DWVFse%vCedZJ5vHb"
    b"Qmzy+KTKu93(5z~D35xBJ`7+G7#>zh3H^w@o2{gqk?4Ty42W9-ja1(G+qC#_@N#Fk&j)pkXxr9nap$H5{t>A@%?vRzgM=Qvc)z<B"
    b"zuf7@+q+#Oz$mjChz(mY(`Mcw(I6ZKFGi++7wiuQEQGt>c3d_FT<Y9|VpzFpIGB9FNVTGiV;r~|yAXyf)^qqw8Gl+a{sHW!xKcv`"
    b"2SWe#5QNQHp76|fZhwcj3fMAOvDpDcNC49{^{YvPuwQ+SAnYTT(=44`$7OCY6!f;(cn`f`^F}wC{oM!P9$4q6RkWx?4lJ=U^|R@|"
    b"Mx`T-W>z6tLRFU7udXqmmfGKC<5;OJA}{JGR3%GL87gbbRo0Baf=FM#W|VJW<Vi}E;>C^zE2yZnIE6Wt=X*ak+Y@OU5UU_hC*9!h"
    b"4!HO>+R{u`$KHr%6iza^C>pZa446`{NDOu8N&_PDkQ$h#0%0matv6e(dIL`VebfDn1Kr`4*zR_RwFkPUt!f;1P*X}IHyeu_#$g5!"
    b"F+CI34gdj-QV&$U#Hxf`ou#4;S3a_sNl-+1j13)Tr7|L58nMwHQQ4!ej4x5A<CMe+Fle4b9QBPSHOc<&&i<%dds5@t$fd=G37xDp"
    b"`rBLE4^JN8OnHgh&C&t5ocI^IvGc5SCpM11$T;e8stV+dQcP_|ij<76mt{9+y2w+rIdN3TPr}HO<|jJ<G~LpOgM!Kq{5c#Ot1cXm"
    b"#Gp&Cs;24-%Ad?Tc={Lp9<aR`OXsqsY+ZD(+?q$}7~F?DGE#ts2;~sjq1kJ*tu}*Z)IkT$9z1MoKF96=BmxUwf}_&3+qmerSr=vu"
    b"Uf>dv1GVG}Ja&=Su(XI`2yn!iJlF!*3AjTCmkn@uQ_v#@7j`DtT?zJHK$ka$>kO--sNVrEhzwfq@M&qDsEw0kz>~lN5eu3eix_3|"
    b"4)%_ctNiENi6}&~x?XU1yNEgX&s$pWDne|z-d4l_`Th)czu{oFuMxW8E0WrGGIsBSdkFS?uFEWft$WZ^Re{+!kTfb$Vqzt>I?YbR"
    b"pX9gAN}AwYsRB9S;PY#B05G*KFv*T`0sA*-_`xjy2iPHwYdrvlo7RVGyTXvxvm%Dx=;ij#U^uL=L(^?lrT8jhhhEK7HLdV-UQYBG"
    b"{0%Qsj^!i^DO}Y=DNyo3%^iLG*tT@&xY`INW~j}Iwl0pDiSjltu9Y`lP6{DRSqUo730MnN44_MW(vhwSLeCly??|1_5P3@HVyI$@"
    b"FNVmaIaXAGH3#`u(}qe~&0uRq)CL#Q)-BjT05;~=Fkn5|N^M6Q9r}Ta6n;;Ehbh=0%WrEc)LN)cLfY+@wlKn+1h07fI)~$y0==fV"
    b"&joUlt`JjITBZU>2WBuuLNy?&cmu~iQN3YT+d&it9ly<(iKHj?#CC0gmTg;awH~oZ&alT;-_}_%zs`gW7HKw&=~Y`11lEj2Agfj^"
    b")=t2I&MP3;SZT4kshd?Zqx6WH{lK5N8DvJCwa^o4{R3h16vJSJQ-B~B)24lk>J%Wgh->UyW(xNr`$~q+8MU?o2RA5!>7x$y@uEjm"
    b"l_DBg3f|pBpam<7?zf+pJne12?20b$)v7nsR#k7iAOR+dN71_hf(&U79k0App<g9VgBqW63eTlP1p{Km!DCDcc<u-sJ720&Rb@Ho"
    b"SeJd8=Abg)_e`Ce$~hDY3>G&2?8@`7LUfipPf-oor0yjz#^^#xv>-WZ+*|ld6jK%3>wume4*B%o+Db5CkDUTFE~rygH{539f)7)w"
    b"NzBsuoJv_Kjl&CAt1NSRM-Z_}RI5cHb>?QC10`|qw&G*}fl7{5f)0_ivu{ig?+<tP_O@$DYTMR}y`1LP;*p}APajVHtV@Erco0T~"
    b"Bs~`^(|8vqig}()D8qCYlW(a0gq>n)0$jAMy*7Kb?G9d_s8D$n-cFnCbh}-E3O{So2#*flhH-eF)plo>^aE7<6v=zQE?m;%86l$f"
    b"g#83=hiY%L4iXc)huSFUxHgy!yD1|29lf)?u&dg<>8744T?V{NCLQ#;)n<S$mGy(}>!7CrN=J3s!or>EStzkaWj~aq);L-Wxf=Fl"
    b"raHltw}NzKILuPdKZSHeHuU-t35F==cOB~9Q9hzYfrZmj@jG?|R;cR+uUl8E35qjyw$oSJ!djodf^awW3bf{~q!)B!kP>)WPxHOu"
    b"aM0TqbwX|1s(N}aV#l_q-mRXhz)3xuntQCRtz>>ZRRkNrj2L@s3gZARX@)U49m7)I+I4@;FwOm?T1kfc)RmQdg!a-^-f1sEV)ZvT"
    b"lU_Rb1H@Qq5?1#o8J990VIeCTO?Gy2)H(co)Mlq=>Nigku&Bwh&3-PhXkMb~H`V>h8uZzR!((+N<in?9(}w-}+RL1R^USqp`D(A6"
    b"bmju^3hr4($tJ3rWdbu^-g4i2wb;twP!CB^yv2}d7zhktU$Y6gpwO_o4Xl0ETk!^bn@!-o-TmFcaHp1iT-!=s1NG=5i4AvULnRBS"
    b"E)ck>RMH))a#h1f`wE#ui~0>W^`jnGP;3Cyt9hW(5U<01bq%IE<XhFY5s9h-bGCa`(^5tSEqpY`QonD~<g|F~Q{B+xDjMwP|05l}"
    b"-YkVQZB<lyt<lHxh>AB+r`tVstt=KZUWi#Bc>{W~XJ}(5&{L&<DtX!pdm`<1$EWhMD^*PqWnh}E2d|7nU(LZh_bPMu!&(Y2;j5Rh"
    b"a$4)+vUjI+&>eyPf{=7+3joz)o4uRzvGRNLt0ui^jzH<*>GHuU9o6|sppj!xSvTO5&u3#;t0^z~T>y*Vo}PvDjA$n5_Fitks#SBg"
    b"t!gFrm1UzlS=FmoVbJ0Q`*VG*s!VwXCXF~FxH7hK-3!&Cy>$ga0$!Vgcm-7d7yWkAlyAp4X!8rHO514D1G+V0??yPIL^GNR6~QcK"
    b"RzBN7u`?UNIwf6WGWoOmOJyc8R9PuFTXrS3Q|z-nE=ZlsNF*NwXk40FW&_mnDB=rC?M=-qNuL;Y(ivTc1Ma}?xG%~B38oVZ2Dcj<"
    b"yKV6FJ1PxDsC9fqH^I;^wBSjhI9vcZC@52Q6*G_E<fzkffe{?yhnJO$g8qd$1dcH}T>_*ej`L>l?5jwza3ZS9_*F7}m5Y;=n~Ion"
    b"(Z}rm4a5XOQM^f6P4Ve4)FQ&(R#YLV$G#;dnklsvWUXwb(N9tl1_U}yDuJc5aqF^qCYNUOfLP4}@%&1f%idD@CXXf=?2NkGd$sq2"
    b"y|&-$(X6RNT2poO^oh<>I`@OtV2)K@fDQcie;PzDVrR;VU}2&BUVt^6Q<7+4zW#c3r^Q9pJ{rBI<v+$nVOqYnAm6@Jra*yd1aAXH"
    b"S!wg^iSk5)*3eK>lV>KKj{ROM4Yl@nT<QzFo0E|~S9MTpZvFOkZ=>xU!2-R%&<WfX-TqHwpx#>4ZRk7j{Z_=@W2Kb6OYP}ApCiAQ"
    b"EYJnin3POx48+Q<DR4(?CBPNlP^?$WdzE?-i&+CObWQ80S#d@YfK$T(gf**J0b${oZc2qx0QiXG<MBp4>d*oAZ|$TbR?m{_5aXPU"
    b"U|(y1tlyW1U_180zhQS(i%P$*cmM9{Nu)jQg*8W7*jxTpg)q6PigDh8&-5`b-(oIY7g_ohJ5+ILueGXd<f@E-X$IF|gfODZ%b7<t"
    b"1>@M(_oV1a^jG^pq{G;KEk8c2>s`-#4|NN`?=EPJtew1`<o@N6x}th2Fzd7XkVIj9(sxl+FC%6z(%(q$V;q&}<NKLyI-YN{KYcp+"
    b"$gah*t%DeJ?;-N?PHEKMW2{|}LH-0&o+?je?i~7&-zIpbF<&IAP35(MYzPcj*{#~q^Wv$&hdLcrJKW^TVqV7GXy-9FUMj8@)BxL6"
    b"ONJ@E>iA=I3iaVopM&U(-;|4|2Vuf77*$N=W5bTnNynU|O*@)|RUtO<eH^}j3<#z)>%m~lvjLsH+N*;yf3^20h%28L|BN3wplq<o"
    b"vDm<sB8Y(tm?AxdzSjVjHP_Y3st8c~Kt(=SfSJKs%^Axx46B_M#W+#}v0o7x3Cz_>enW)?$}v$spjk(}UJSG(kq}wYakh0tb!jzS"
    b"=#bch>X(W_Pi(e!2Ymwl&KZ&l9x>_@Mc|!XHoh{IHfLX*g8Ay{m#^$+$|MIbAuhqVn=VXz4K*r2>~uQ#f8YL4f4F|0axCz38i1{<"
    b"ibw6u;I*qr=`noO)5Dp)9M3BCJ}+RnXP2SH&hMIhq`R3Y@o3aVSmmk`?_{M9ok=HNV?V+CI?H%sU4vk>kJ$q}n2fq?gEcwpupF;~"
    b"@&Wj5k6o^QZ2Eus*6DMYo>!({b!Y;jMSPB*HfehAD&vs!rdQP<e`WVX;s;>6`@A2$8qr_e{IzPr@72bsn>T0w$7pNx@W(2bdP_0M"
    b"MiW~%wV3>fljtKlTxzTJy>0q7hTWOEQQ#*Fv%LJSwwy3YQk0frdeOZim4kn;ml(zTOI07r3%pd*%*0ITG8tSj%*u$$QZQ4iYcTiQ"
    b"Y^UX_^#;6{G(l49ImPu`mT7V=v8SHKw(z?f%nspYNBn)84$H|wud)hNX4v!?Kj3F79J4yE`Nx(`j>A}$SsyvU2)(R-{QG~cb@Fid"
    b"IN{A!l+Iv^GG1~BHvDe_5W3rV>bVeza)C8Lm?mqk!H9R20XILJ<t6rB;dz3nS=>KVA+tk`ZI-se_jgN-U=Esls1wwZemY=Xr5HMU"
    b"ZPu*}!0uE_C=bAu9*yw8mQJ>0b;=tKJ@DL<{Gp&}f=!ti>9fA-yb$<f!p$F5A?}nHKo{A(JLpEeol4`=RifWdYIndBHy>iE=20jy"
    b"ZE`5^&&9TsGeMrJ&7+%q%J-6Zz;Io_tWrP!7x8fgQL#HJy9k-~nIS`kMW1w-SNlKTH}%F=T!I(u1btFW(VGqF3RhCBKy?U;3v)uJ"
    b"lkcGAg?1i)7iOHx6JGBL-65QK4am^Yk65Tq@L$ELTxnnHWUlG-R4U|&AuAaX)4#QjAe8^9-F$_wNhPhi!!Bh2?!t=~^r4zn0cU)S"
    b"K|#4H!{B4Aq!(7)>^EwZ?`|tS;k~GZVFvonxf<!cS-XH0ue@J%%R}^rSGSa#?mVKE87fX&%^J{PIr9Es7~+R3a<<oac4lDUvp><-"
    b"o5Zrv^JWSCf8Dtrqu(ZWHK<+he=4uDlHbI`#2VdRz5H0ofK~a}egRgWXAT5@#ZS~`!dKF16j-aTHmOf4s>47h3ogz!-#7I<@2Ah@"
    b"_qIm+FSqt;&*g2~%BMoRc%lrR#~eRzYCqx&5qy#@WKhx832z>_&)K@tGTePqw()DA(}e~H-krLBp>Hu~xfJwWIQLCRdoe<lBhay{"
    b"yI@fF1zr6iO#G;&s&->?iiX3Se`4B6T|K5N8|E}VJaHBC$lsklSohDrD@k@oT|E4D>yjBdQrXh@&GNB&xk{N@umo+5!_o|RoKiN0"
    b"u6@gN3UNn)0qd<0YROn&t<yNG*#?DTH1IxBEg98UZLu^_xr(~d0`oQ>4v*XBbC!tA0Mlgw;?@Db7G;<|vQ=_<9qh&}g%V13Jvbb{"
    b"zf~ZT5eJ>hhHV7{m4h2mdAeq>364ptQaLDtC7l2cVS1l<FrL0_Yf=F4YGY#$FO}e=`);LrRT~>>$3hzs|J%tZAVCU-3f}SC$t8op"
    b"WkMC1m>w)RUjIqjY_OmUnLyXYc#PesC}PdGM(Lv_+<s{Q`9IYzHE1DV)8J;2T9~Pwc=UEqRF(2NciYumNK$`nP=HsPIhNFzB-|r;"
    b"<YKNq8=}6q7HCq9URQ9D&WRs^uo-?pU_8-$MlqkdjcPAc*^dOM^AS!L8|>FAWIifyse}8xM#u$KwT{;W)LQjDfr)rVk%+oA5qyTB"
    b"ZNFRqlu-AaI+ljCHue2qtnN_X@OGCv)O{rH@`An8p_tvTouPzkwF&^Yc=d1<YJbo>00ze81(*RqkzcNX<zeu4?FZ25$i@^3VG##d"
    b"k81&(nPIX6A!sWcYe=dNgJDgGexpA7LLZUO_-wr3W4z%w%5g9jpiEN~Qo1+ftteH)0F3@hOQZkt0dF>DBDNn-hkns~iPjF7Qrkp~"
    b"qBie3c6hJ~d>(+Jp=<*X3h7<=t@GuWm?h4{9krx(`42c&$yTGk?H*!a5k&3I!~X*QQ%bP"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build from an existing polished notebook when possible, or from "
            "the embedded current template when no suitable notebook exists."
        )
    )
    template_group = parser.add_mutually_exclusive_group()
    template_group.add_argument(
        "--source",
        type=Path,
        help="use this notebook explicitly instead of searching",
    )
    template_group.add_argument(
        "--from-scratch",
        action="store_true",
        help="skip notebook discovery and use the embedded fallback template",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"generated notebook (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--repo-slug",
        default=DEFAULT_REPO_SLUG,
        help="GitHub OWNER/REPOSITORY used in notebook links",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="retain outputs when an existing executed notebook is the template",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing generated output file",
    )
    return parser.parse_args()


def cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def validate_notebook(notebook: dict[str, Any], label: str) -> None:
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{label}: expected notebook format 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{label}: notebook has no cells")

    notebook_text = "\n".join(cell_text(cell) for cell in cells)
    missing = [heading for heading in REQUIRED_SECTIONS if heading not in notebook_text]
    if missing:
        raise ValueError(f"{label}: missing required sections: {', '.join(missing)}")

    for index, cell in enumerate(cells):
        if cell.get("cell_type") not in {"markdown", "code", "raw"}:
            raise ValueError(f"{label}: unsupported cell type at index {index}")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise ValueError(f"{label}: error output found in code cell {index}")


def read_notebook(path: Path) -> dict[str, Any]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read notebook: {exc}") from exc
    validate_notebook(notebook, str(path))
    return notebook


def embedded_notebook() -> dict[str, Any]:
    try:
        raw = zlib.decompress(base64.b85decode(EMBEDDED_TEMPLATE_B85))
    except (ValueError, zlib.error) as exc:
        raise RuntimeError(f"embedded notebook template is damaged: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EMBEDDED_TEMPLATE_SHA256:
        raise RuntimeError("embedded notebook template checksum does not match")
    notebook = json.loads(raw.decode("utf-8"))
    validate_notebook(notebook, "embedded fallback")
    return notebook


def discover_notebook(output: Path) -> tuple[dict[str, Any], str] | None:
    candidates = [DEFAULT_SOURCE]
    candidates.extend(sorted(ROOT.glob("*.ipynb")))
    candidates.extend(sorted(ROOT.rglob("*.ipynb")))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or resolved == output or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            relative = resolved.relative_to(ROOT)
        except ValueError:
            continue
        if any(part in EXCLUDED_SEARCH_PARTS for part in relative.parts[:-1]):
            continue
        try:
            return read_notebook(resolved), str(resolved)
        except ValueError as exc:
            print(f"Skipping invalid notebook {resolved}: {exc}", file=sys.stderr)
    return None


def select_template(
    explicit_source: Path | None, from_scratch: bool, output: Path
) -> tuple[dict[str, Any], str]:
    if from_scratch:
        return embedded_notebook(), "embedded fallback (--from-scratch)"

    if explicit_source is not None:
        source = explicit_source.expanduser().resolve()
        if source == output:
            raise SystemExit(
                "Refusing to use the output file as its own source. "
                "Choose a different --source or --output."
            )
        if not source.is_file():
            raise SystemExit(f"Source notebook not found: {source}")
        try:
            return read_notebook(source), str(source)
        except ValueError as exc:
            raise SystemExit(f"Invalid source notebook: {exc}") from exc

    discovered = discover_notebook(output)
    if discovered is not None:
        return discovered
    return embedded_notebook(), "embedded fallback (no suitable notebook found)"


def replace_repo_slug(source: Any, new_slug: str) -> Any:
    if isinstance(source, list):
        return [part.replace(DEFAULT_REPO_SLUG, new_slug) for part in source]
    if isinstance(source, str):
        return source.replace(DEFAULT_REPO_SLUG, new_slug)
    return source


def remove_transient_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(metadata)
    cleaned.pop("execution", None)
    cleaned.pop("executionInfo", None)
    cleaned.pop("outputId", None)
    colab = cleaned.get("colab")
    if isinstance(colab, dict):
        colab.pop("base_uri", None)
        if not colab:
            cleaned.pop("colab", None)
    return cleaned


def build_notebook(
    source_notebook: dict[str, Any], repo_slug: str, keep_outputs: bool
) -> dict[str, Any]:
    notebook = copy.deepcopy(source_notebook)

    for cell in notebook["cells"]:
        cell["source"] = replace_repo_slug(cell.get("source", ""), repo_slug)
        cell["metadata"] = remove_transient_metadata(cell.get("metadata", {}))
        if cell.get("cell_type") == "code" and not keep_outputs:
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook["metadata"] = remove_transient_metadata(notebook.get("metadata", {}))
    colab = notebook["metadata"].setdefault("colab", {})
    colab["name"] = NOTEBOOK_NAME
    colab["provenance"] = []
    if not keep_outputs:
        colab.pop("gpuType", None)
        language_info = notebook["metadata"].get("language_info")
        if isinstance(language_info, dict):
            language_info.pop("version", None)

    return notebook


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo_slug):
        raise SystemExit("--repo-slug must have the form OWNER/REPOSITORY")

    output = args.output.expanduser().resolve()
    if output == DEFAULT_SOURCE:
        raise SystemExit(
            "Refusing to overwrite the canonical submitted notebook. "
            "Choose a different --output path."
        )
    if output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output}\nRerun with --force to replace it.")

    source_notebook, template_label = select_template(
        args.source, args.from_scratch, output
    )
    notebook = build_notebook(source_notebook, args.repo_slug, args.keep_outputs)
    validate_notebook(notebook, str(output))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    temporary.replace(output)

    code_cells = sum(cell.get("cell_type") == "code" for cell in notebook["cells"])
    mode = "with outputs" if args.keep_outputs else "source-only (outputs cleared)"
    print(f"Template: {template_label}")
    print(f"Built: {output}")
    print(f"Validated: {len(notebook['cells'])} cells ({code_cells} code), {mode}.")
    print(f"GitHub links: https://github.com/{args.repo_slug}")


if __name__ == "__main__":
    main()
