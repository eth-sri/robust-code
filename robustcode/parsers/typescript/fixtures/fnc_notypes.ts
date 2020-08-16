function add(x, y) {
    if (typeof x === "number") {
        return 5;
    }
    let a = 3.0;
    return x + y + a;
}

let myAdd =
    function(x, y) { return x + y; };

function buildName(firstName = 'd', bar = "Smith", lastName?) {
    return firstName;
}

interface UIElement {
    addClickListener(onclick);
}

class Handler2 {
    info;
    onClickGood(this, e) {
        // can't use `this` here because it's of type void!
        console.log('clicked!');
        let c = null;
        c = undefined;
        c = this;
    }
}

class Handler {
    info;
    constructor(theName) { this.info = theName; }

    onClickGood = (e) => {
        this.info = e.message;
        return {
            foo: 'bar',
            bar: Handler2
        }
    }

    public add(operand) {
        return this;
    }
}

function pickCard(x);

interface MyArray<T> {
 reverse();
}

let handler = new Handler();

abstract class Animal {
    readonly name;
    constructor(theName) { this.name = theName; }
    move(distanceInMeters = 0) {
        console.log(`${this.name} moved ${distanceInMeters}m.`);
    }
    abstract makeSound()
}

class Snake extends Animal {
    constructor(name) { super(name); }
    move(distanceInMeters = 5) {
        console.log("Slithering...");
        super.move(distanceInMeters);
    }
    makeSound() {}
}

function isNumber(x) {
    return typeof x === "number";
}