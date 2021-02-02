var SVG = "http://www.w3.org/2000/svg";

function points_to_string(pts, x, y, a, b, theta) {
    theta = -theta + Math.PI; 
    var sin_theta = Math.sin(theta);
    var cos_theta = Math.cos(theta);
    str = ""
    for(var i in pts) {
	var pt = [pts[i][0] * a, pts[i][1] * b]
	str += (x + pt[0] * cos_theta - pt[1] * sin_theta) + "," + (y + pt[0] * sin_theta + pt[1] * cos_theta) + " ";
    }
    return str;
}


function two_digit_hex(r) {
    var s = Math.floor(r).toString(16);
    return s.length < 2 ? "0" + s : s;
}

function get_color(r, g, b) {
    return "#" + two_digit_hex(r) + two_digit_hex(g) + two_digit_hex(b);
}


class Fly {
    static triangle_pts = [[.5,-.5], [.5,.5], [-.5, 0]];
    static left_wing_pts = [[0,0], [1,0]];
    static right_wing_pts = [[0,0], [1,0]];
    static num_path_frames = 100;
    static colors = ['Aqua', 'BlueViolet', 'Coral', 'DarkGoldenRod', 'DarkGreen', 'DarkMagenta', 'DarkSlateGray', 'LightGreen', 'CadetBlue', 'LightSteelBlue', 'MediumOrchid', 'MistyRose', 'Olive', 'SaddleBrown', 'Yellow', 'RosyBrown', 'PaleVioletRed', 'Wheat', 'YellowGreen', 'DarkKhaki'];
    
    constructor(fly_ind, chamber, female, type, data, sample_ind=0, past=null) {
	this.fly_ind = fly_ind;
	this.female = female;
	this.type = type;
	this.chamber = chamber;
	this.data = data;
	this.sample_ind = sample_ind;
	this.is_best = false;
	this.is_visible = true;
	this.path_color = type == "future" ? [128,128,128] : (this.female ? [255,200,200] : [0, 0, 255]);
	var e = chamber.example_ind;
	this.t_start = past ? past["x"][0].length : 0;
	this.past = past;
	this.style = (female ? "female" : "male") + " " + this.type;
	this.initialize_svg();
    }

    initialize_svg() {
	var svg = SVG;
	var color = Fly.colors[this.fly_ind % Fly.colors.length];
	this.group = document.createElementNS(svg, 'g')
	this.triangle = document.createElementNS(svg, 'polygon');
	this.left_wing = document.createElementNS(svg, 'polyline');
	this.right_wing = document.createElementNS(svg, 'polyline');
	this.track = document.createElementNS(svg, 'polyline');
	this.path = [];
	this.path_group = document.createElementNS(svg, 'g');
	for(var i = 0; i < Math.min(Fly.num_path_frames, this.chamber.T) + 1; i++) {
	    var e = document.createElementNS(svg, 'line');
	    e.setAttribute("class", "track " + this.style);
	    this.path.push(e);
	    this.path_group.appendChild(e);
	}
	this.chamber.svg.appendChild(this.path_group);
	this.triangle.setAttribute("class", " triangle " + this.style);
	this.left_wing.setAttribute("class", this.style + " left_wing");
	this.right_wing.setAttribute("class", this.style + " right_wing");
	this.triangle.setAttributeNS(null, "stroke", color);
	//this.triangle.setAttribute("stroke", color);
	this.group.appendChild(this.triangle);
	this.group.appendChild(this.left_wing);
	this.group.appendChild(this.right_wing);
	this.group.appendChild(this.track);
	this.chamber.svg.appendChild(this.group);

	this.group.fly = this;
	this.group.onclick = function() {
	    var fly = this.fly;
	    fly.chamber.clicked = fly.chamber.selected = fly;
	    fly.chamber.synch_visible();
	};
	this.group.onmouseover = function() {
	    var fly = this.fly;
	    var old = fly.chamber.selected;
	    if(old != fly) {
		fly.old_selected = old;
		fly.chamber.selected = fly;
		fly.chamber.synch_visible();
	    }
	};
	this.group.onmouseleave = function() {
	    var fly = this.fly;
	    fly.chamber.selected = fly.chamber.clicked ? fly.chamber.clicked : fly.old_selected;
	    fly.old_selected = null;
	    fly.chamber.synch_visible();
	};
	this.group.setAttribute('class', 'svgGroup');
    }
    
    set_position(x, y, theta, a, b, lw_a, lw_l, rw_a, rw_l) {
	this.triangle.setAttributeNS(null, 'points', points_to_string(Fly.triangle_pts, x, y, a, b, theta));
	this.left_wing.setAttributeNS(null, 'points', points_to_string(Fly.left_wing_pts, x, y, lw_l, b, theta + lw_a));
	this.right_wing.setAttributeNS(null, 'points', points_to_string(Fly.right_wing_pts, x, y, rw_l, b, theta + rw_a));
    }

    update_path() {
	var t = this.chamber.t;
	var c1 = this.chamber.background_color;
	var c2 = this.path_color;
	var e = this.chamber.example_ind;
	var xs = this.data["x"][this.sample_ind];
	var ys = this.data["y"][this.sample_ind];
	if(this.past) {
	    xs = this.past["x"][0].concat(xs);
	    ys = this.past["y"][0].concat(ys);
	}
	var f = this.fly_ind;
	var j = 0;
	var s = Math.max(1, t - Fly.num_path_frames);
	var e = Math.min(t, xs.length - 1);
	if(!this.is_visible)
	    e = -1;
	for(var i = s; i <= e; i++, j++) {
	    //var w2 = 1 - (t - i) / Fly.num_path_frames;
	    //var w1 = (t - i) / Fly.num_path_frames;
	    //var color = get_color(c1[0]*w1 + c2[0]*w2, c1[1]*w1 + c2[1]*w2, c1[2]*w1 + c2[2]*w2);
	    var color = i >= this.t_start ? Fly.colors[this.fly_ind % Fly.colors.length] : 'Gray';
	    
	    this.path[j].setAttributeNS(null, 'x1', xs[i][f])
	    this.path[j].setAttributeNS(null, 'y1', ys[i][f])
	    this.path[j].setAttributeNS(null, 'x2', xs[i-1][f])
	    this.path[j].setAttributeNS(null, 'y2', ys[i-1][f])
	    this.path[j].setAttributeNS(null, "stroke", color);
	    this.path[j].setAttributeNS(null, 'visibility', "visible");
	}
	for(var i = 0; i < this.path.length; i++)
	    if(i < s || i > e)
		this.path[i].setAttributeNS(null, 'visibility', "hidden");
    }

    refresh() {
	var t = this.chamber.t - this.t_start;
	var data = this.data;
	var f = this.fly_ind;
	var s = this.sample_ind;
	if(t >= 0 && t < data['x'][s].length && this.is_visible) {
	    this.set_position(data['x'][s][t][f], data['y'][s][t][f], data['theta'][s][t][f], data['a'][s][t][f], data['b'][s][t][f],
			      data['l_wing_ang'][s][t][f], data['l_wing_len'][s][t][f], data['r_wing_ang'][s][t][f], data['r_wing_len'][s][t][f]);
	    this.group.setAttributeNS(null, "visibility", "visible");
	} else {
	    this.group.setAttributeNS(null, "visibility", "hidden");
	}
	this.update_path();
    }
}

class Chamber {
    constructor(svg_id, data, example_ind=0) {
	this.show_future = true;
	this.show_past = true;
	this.show_all_simulated = false;
	this.show_first_simulated = false;
	this.show_best_simulated = true;
	this.clicked = this.selected = null;
	this.data = data;
	this.male_inds = data['male_inds'];
	this.female_inds = data['female_inds'];
	this.chamber = data['chamber'];
	this.svg = document.getElementById(svg_id);
	this.example_ind = example_ind;
	var e = example_ind;
	this.t = "past" in data ? data["past"][e]["x"][0].length : 0;
	this.T = "past" in data ? data["past"][e]["x"][0].length + data["future"][e]["x"][0].length : data["simulated"][e]["x"][0].length;
	this.background_color = [0, 0, 0];
	this.initialize_svg();
	this.initialize_flies();
	this.synch_visible();
    }

    set_frame(t) {
	this.t = t;
	this.refresh();
    }
    
    initialize_svg() {
	this.chamber_svg = document.createElementNS(SVG, 'polygon')
	this.chamber_svg.setAttribute("class", "chamber");
	var pts = "";
	for(var i = 0; i < this.chamber[0].length; i++) {
	    pts += this.chamber[0][i] + "," + this.chamber[1][i] + " ";
	}
	this.chamber_svg.setAttributeNS(null, 'points', pts);
	this.svg.appendChild(this.chamber_svg);
	this.svg.setAttribute("viewBox", Math.min(...this.chamber[0]) + " " + Math.min(...this.chamber[1]) +
			      " " + Math.max(...this.chamber[0]) + " " + Math.max(...this.chamber[1]));
    }
    
    initialize_flies() {
	this.flies = [];
	var types = ['future', 'past', 'simulated'];
	for(var t in types) {
	    for(var i = 0; i < this.male_inds.length + this.female_inds.length; i++) {
		var fly_ind = i < this.male_inds.length ? this.male_inds[i] : this.female_inds[i - this.male_inds.length];
		var female = i < this.male_inds.length ? false : true;
		var data = this.data[types[t]][this.example_ind];
		var past = types[t] != 'past' && 'past' in this.data ? this.data['past'][this.example_ind] : null;
		var best = null;
		var best_dist = null;
		for(var s = 0; s < data['x'].length; s++) {
		    var f = new Fly(fly_ind, this, female, types[t], data, s, past);
		    if(types[t] == 'simulated' && 'future' in this.data) {
			var future = this.data['future'][this.example_ind];
			var ti = data['x'][s].length - 1;
			var dx = data['x'][s][ti][fly_ind] - future['x'][0][ti][fly_ind];
			var dy = data['y'][s][ti][fly_ind] - future['y'][0][ti][fly_ind];
			var dist = dx*dx + dy*dy;
			if(!best || dist < best_dist) {
			    best = f;
			    best_dist = dist;
			}
		    }
		    this.flies.push(f);
		}
		if(best)
		    best.is_best = true;
	    }
	}
    }

    refresh() {
	for(var f in this.flies)
	    this.flies[f].refresh();
    }

    synch_visible() {
	for(var i in this.flies) {
	    var f = this.flies[i];
	    var show = (this.show_past && f.type == 'past') || (this.show_future && f.type == 'future') ||
		(f.type == 'simulated' && (this.show_all_simulated ||
					   (this.show_first_simulated && f.sample_ind == 0) ||
					   (this.show_best_simulated && f.is_best)));
	    var selected = this.selected && this.selected.fly_ind == f.fly_ind;
	    if(this.selected) {
		if(selected) {
		    var x = 1;
		}
	    }
	    f.is_visible = show || selected;
	}
	this.refresh();
    }
}
