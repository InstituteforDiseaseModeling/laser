cat pop_1M_25nodes_seeded.csv |
    awk -F, '{
        age=; node=;
        if( age>15 ) {
            int_age=int(age); mcw[node_int_age] ++
        } else {
            print -bash
        }
    } END {
        for ( bin in mcw ) {
            split( bin, parts, _ );
            print ?,parts[1],parts[2],mcw[bin]
        }
    }'
