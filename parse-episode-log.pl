#!/usr/bin/perl -w
my $ep0 = 0;
if (@ARGV >= 2 && ($ARGV[0] eq "--base" || $ARGV[0] eq "-b")) {
    shift;
    $ep0 = shift;
} else {
    print "episode,reward,rate,duration,steps,entropy,policy_loss,value_loss\n";
}
my $prev_ep = -1;
while (<>) {
    if (/episode (\d+) at \d+ steps: reward=([-\d.]+) rate=([-\d.]+) duration=([-\d.]+) steps=([-\d.]+).*[}] (entropy=([-\d.]+) policy_loss=([-\d.]+) value_loss=([-\d.]+))?/) {
        my $ep = $1 + $ep0;
        if ($ep <= $prev_ep) {
            $ep = $prev_ep + 1;
            $ep0 = $ep - $1;
            print STDERR "Adjusting episode number from $1 to $ep, offset $ep0\n";
        }
        my ($x_e, $x_p, $x_v) = ('', '', '');
        if (defined $6) {
            ($x_e, $x_p, $x_v) = ($7, $8, $9);
        }
        print "$ep,$2,$3,$4,$5,$x_e,$x_p,$x_v\n";
        $prev_ep = $ep;
    }
}
