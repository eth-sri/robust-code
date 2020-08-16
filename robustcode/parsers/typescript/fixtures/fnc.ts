function add(x: number, y: number): number {
    if (typeof x === "number") {
        return 5;
    }
    let a = 3.0;
    return x + y + a;
}

let myAdd: (x: number, y: number) => number =
    function(x: number, y: number): number { return x + y; };

function buildName(firstName: string = 'd', bar = "Smith", lastName?: string) {
    return firstName;
}

interface UIElement {
    addClickListener(onclick: (this: Koo[], e: Event[]) => void): void;
}

class Handler2 {
    info: string;
    onClickGood(this: void, e: Event) {
        // can't use `this` here because it's of type void!
        console.log('clicked!');
        let c = null;
        c = undefined;
        c = this;
    }
}

class Handler {
    info: string;
    constructor(theName: string) { this.info = theName; }

    onClickGood = (e: Event) => {
        this.info = e.message;
        return {
            foo: 'bar',
            bar: Handler2
        }
    }

    public add(operand: number): this {
        return this;
    }
}

function pickCard(x: {suit: string; card: number; }[]): number | string;

interface MyArray<T> {
 reverse(): Foo<T>[];
}

let handler = new Handler();

abstract class Animal {
    readonly name: string;
    constructor(theName: string) { this.name = theName; }
    move(distanceInMeters: number = 0) {
        console.log(`${this.name} moved ${distanceInMeters}m.`);
    }
    abstract makeSound(): void
}

class Snake extends Animal {
    constructor(name: string) { super(name); }
    move(distanceInMeters = 5) {
        console.log("Slithering...");
        super.move(distanceInMeters);
    }
    makeSound(): void {}
}

function isNumber(x: any): x is number {
    return typeof x === "number";
}
